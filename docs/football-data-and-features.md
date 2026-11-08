# NFL Football Data Contracts & Feature Engineering Specification

> **Related Documents**: [System Architecture Blueprint](./architecture.md) | [Hockey Data & Feature Specification](./hockey-data-and-features.md) | [EPL Soccer Data & Feature Specification](./soccer-data-and-features.md) | [MLB & NPB Baseball Data & Feature Specification](./baseball-data-and-features.md) | [Implementation Roadmap](./roadmap.md)

The football plugin (`src/sports_ml/sports/football/`, `--sport football --league nfl`) models the **National Football League** under the shared two-stage architecture. `nfl` is the only implemented league and the default; the `league` column and league-partitioned registries keep `ncaaf` reserved (the adapter rejects it as unimplemented) so a college plugin can be added without restructuring.

---

## 1. NFL Structural Constraints

| Dimension | NFL Reality | Architectural Handling |
| :--- | :--- | :--- |
| **Sample size** | 17 games/team/season (16 before 2021), ~272 games/season | Heavy Bayesian shrinkage with cross-season carry-over; calibrators fit on pooled prior seasons with parametric (Platt/temperature) calibration, not isotonic |
| **Starting QB** | Dominates outcomes | QB identity is first-class: QB-adjusted Elo (`FootballQBEloHook`), QB-level priors following the player, team-level priors following the roster/scheme |
| **Weekly cadence** | One game per team per week, byes, Thursday/Monday/international/neutral games, playoff bracket `WC`→`DIV`→`CON`→`SB` | `event_seq == 1` always; one game per team per `(season, week)`; ordering key `(start_time_utc, event_seq, game_id)` |
| **Ties** | ~0.2% of games | **2-way binary**: ties are `0.5` in Elo updates and are **excluded** from classifier targets and calibration (never a third class) |
| **Season key** | Season starts in September, ends in February | `season` is the 4-digit start year (`2024` includes Feb 2025 playoffs and Super Bowl), matching nflverse |
| **Team relocation** | `OAK→LV`, `SD→LAC`, `STL→LA` | Aliases resolve to the current tricode so rating history follows the franchise (`FRANCHISE_ERAS`) |

### 1.1 Leakage Traps in nflverse

Post-game or observed schedule columns must never enter feature tables:
* `spread_line`, `total_line`, `home_moneyline`, `away_moneyline` → evaluation-only `market_*` columns.
* `temp`, `wind`, observed `roof` → observed-vs-forecast handling (Milestone 27; `obs_weather_*` namespace).
* `home_score`, `away_score`, post-game inactives/injury designations → label side only.

---

## 2. Data-Source Map & Availability Windows

All sources are nflverse release assets consumed **directly** as parquet / `csv.gz` (optionally via `nflreadpy`). The deprecated `nfl_data_py` package is **not** a dependency.

| Asset | Used For | Availability |
| :--- | :--- | :--- |
| `schedules/games` | Schedule, kickoff time, venue, scores (labels), quarantined market columns | 1999+ |
| `pbp_{YYYY}` | EPA/WP, success rate, QB identity (first offensive snap), team game aggregates | 1999+ |
| `player_stats_{YYYY}` (weekly) | Stage 2 player logs | 1999+ |
| `rosters` / `rosters_weekly`, `players` | ID crosswalk (`gsis_id` canonical; `pfr_id`, `espn_id` mapped) | 1999+ |
| `depth_charts_{YYYY}` | Tier 2 starting-QB projection | ~2001+ |
| `injuries_{YYYY}` | Practice/game-status availability | ~2009+ |
| `snap_counts_{YYYY}` | Snap share, roster-continuity weighting | 2012+ |
| `nextgen_stats` | Passing/rushing/receiving tracking features | 2016+ |
| `draft_picks` | Rookie/debut QB priors | 1999+ |

Missing-source degradation is explicit via `has_snap_counts`, `has_ngs`, and `has_injury_report` flags (Milestone 23).

### 2.0 Ingestion (`sports/football/ingestion/`)
* `nflverse_client.NflverseClient`: retrying (`tenacity` exponential backoff; 404 is terminal) `httpx` client. Raw caches live in `data/raw/football/nflverse/` (`*.parquet` natively compressed; schedule CSV Gzip-compressed through `core/cache.py` as `schedules.csv.gz`). Completed seasons cache permanently, the active season refreshes after a TTL (6 h), and a failed refresh falls back to the stale cache. Assets outside their availability window (e.g., injuries before 2009) return `None` without a network call.
* `ingestor.NflverseIngestor`: `--from-year`, `--current-season`, and `--start-date/--end-date` sync; idempotent upserts by canonical keys so corrected weeks overwrite; `has_snap_counts`, `has_ngs`, `has_injury_report` flags and `missing_assets` degrade gracefully.
* `reconciler.FootballSilverReconciler`: builds the Silver tables above with `zstd` Parquet; `health.check_football_data_sources` probes every asset for reachability and its required-column contract (`sports-ml check-sources --sport football --league nfl`; `sports-ml ingest --sport football --league nfl ...`).
* Schedule `gametime` is Eastern Time (`America/New_York`) and defaults to 13:00 when missing; `stadium_id` maps to the stadium registry (`NFLVERSE_STADIUM_ID_TO_VENUE`), with unlisted legacy venues proxied by the host club's current home venue.

### 2.1 Pre-Game Availability Tiers (defined in Milestone 23)
* **Tier 1 (Official, ≈90 min pre-kickoff)**: official inactives + confirmed starting QB.
* **Tier 2 (Early-week projection)**: depth chart + Wed–Fri practice designations → probabilistic starting QB / `p_starts`; never uses post-game inactives.
* Every snapshot carries `as_of_time_utc`; `FootballStartingQBResolver` only reads rows with `as_of_time_utc <= min(requested as-of, kickoff − 90 min)`.
* **Tier 1** (`resolve_tier1`): confirmed starter (first-dropback QB from play-by-play, stamped at the cutoff as a historical proxy for the official confirmation; live mode supplies as-of snapshots), else depth chart minus the official inactives; falls back to Tier 2 when neither exists.
* **Tier 2** (`resolve_tier2`): QB depth cascade `P(QB_i starts) = a_i * Prod_{j<i}(1 - a_j)` (normalized) with `a_j = 0.97 * m_j`, where `m_j` is `0 / 0.05 / 0.60` for `Out / Doubtful / Questionable` (else `0.85 / 0.95 / 1.0` by practice participation). Tier 2 reads only the `depth_chart` and `injury_report` sources, never `inactives`.

---

## 3. Canonical Polars Parquet Schemas (Silver Layer)

Silver tables live under `data/silver/football/`, sorted by `(start_time_utc, event_seq, game_id)`. The Python source of truth is `src/sports_ml/sports/football/constants.py` (`SILVER_FOOTBALL_*_SCHEMA`); Milestone 23 extends team/player logs with play-by-play aggregates.

### 3.1 `games.parquet` (1 row per game)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `sport` | `pl.Utf8` | Always `"football"` |
| `league` | `pl.Utf8` | `"NFL"` (`"NCAAF"` reserved, rejected) |
| `season` / `week` | `pl.Int32` / `pl.Int16` | Start-year season; week continues through playoffs |
| `season_type` | `pl.Utf8` | `"REG"`, `"WC"`, `"DIV"`, `"CON"`, `"SB"`; `"PRE"` is rejected from model tables |
| `game_id` | `pl.Int64` | Canonical unique id (`event_id` alias) `SSSSWWAAHH` = season, week, away-team index, home-team index (`make_football_game_id`). Derived purely from the nflverse string id (`2023_01_BUF_NYJ`) after franchise normalization, so every nflverse table (schedule, play-by-play, snap counts) joins without a lookup |
| `nflverse_game_id` | `pl.Utf8` | Original nflverse string key (crosswalk/audit only) |
| `event_seq` | `pl.Int32` | Always `1` |
| `game_date`, `start_time_utc` | `pl.Date`, `pl.Datetime("us","UTC")` | Local date; kickoff instant (canonical sort key) |
| `home_team`, `away_team` | `pl.Utf8` | Canonical nflverse tricode (`LA`, `LAC`, `LV`, `WAS`, `JAX`, …) |
| `venue_id`, `venue_timezone` | `pl.Utf8` | Stadium registry key and IANA timezone |
| `is_neutral_site`, `is_international` | `pl.Boolean` | Super Bowl / neutral / international venue flags |
| `game_state` | `pl.Utf8` | `"FINAL"` or `"FUT"` |
| `home_qb_id`, `away_qb_id` | `pl.Int64` | Starting QB (`gsis_id` stripped of the `00-` prefix, via `gsis_id_to_int`) |
| `overtime` | `pl.Boolean` | Game went to overtime |
| `home_score`, `away_score`, `home_win`, `is_tie` | ints | Labels only; ties have `home_win` null and `is_tie = 1` |
| `market_*` | `pl.Float64` | Evaluation-only: `market_home_decimal_odds`, `market_away_decimal_odds`, `market_prob_home`, `market_prob_away` (power de-vigged moneylines), `market_spread_home` (`-spread_line`: negative = home favored), `market_total_points` |
| `obs_weather_*` | `pl.Float64` / `pl.Utf8` | Observed game-day `obs_weather_temp`, `obs_weather_wind`, `obs_weather_roof`, `obs_weather_surface` (never features; Milestone 27 handles forecast-vs-observed) |

Validation (`FootballAdapter.validate_event_schema`) enforces: known `league`, `event_seq == 1`, unique `game_id`, valid `season_type` (no `PRE`), and **at most one game per team per `(season, week)`**.

### 3.2 `team_game_logs.parquet`, `player_game_logs.parquet`, `qb_game_logs.parquet`
See `SILVER_FOOTBALL_TEAM_LOGS_SCHEMA`, `SILVER_FOOTBALL_PLAYER_LOGS_SCHEMA`, `SILVER_FOOTBALL_QB_LOGS_SCHEMA`.
* **Team logs** (aggregated from play-by-play over scrimmage `pass`/`run` plays, no garbage-time filter): `off_plays`, `off_dropbacks`, `off_rushes`, `off_epa_total`, `off_pass_epa_total`, `off_rush_epa_total`, `off_success_plays`, `off_yards`, `off_turnovers`, `off_sacks_taken`, mirrored `def_*` (the opponent's offense), `penalties`, `penalty_yards`, `time_of_possession_s` (sum of unique drive TOP), plus points and `overtime`.
* **Player logs** (weekly player stats + snap share): the nflverse `gsis_id` is the canonical key (`player_id` = digits of the id); `snaps`/`snap_share` come from `snap_counts` via the `players` `pfr_id -> gsis_id` crosswalk and degrade to null when unavailable.
* **QB logs** (dropback level, `id` column): `dropbacks`, `pass_attempts`, `qb_epa_total`, `cpoe_total`/`cpoe_plays`, `sacks`, `air_yards_total`, `scrambles`, and `is_starter` (first dropback QB).

Uniqueness: `(game_id, team)`, `(game_id, player_id)`, `(game_id, team, qb_id)` (`FootballAdapter.validate_team_game_logs` / `validate_participant_schema`).

### 3.3 `team_week_availability.parquet` (as-of snapshots)
One row per `(game_id, team, player_id, source, as_of_time_utc)` with `source` $\in \{\text{injury\_report}, \text{depth\_chart}, \text{inactives}\}`, `report_status`, `practice_status`, `depth_rank`, and `is_inactive`. Stamps are conservative (a later stamp can only hide information):
* Injury reports use `date_modified`, else `kickoff - 24 h`.
* Depth charts use `dt` when present, else `kickoff - 2 days`.
* Official inactives are reconstructed from weekly rosters (`status == "INA"`) at `kickoff - 90 min`.

---

## 4. Registries (`constants.py`)

* **32-team registry** (`NFL_TEAMS_32`): AFC/NFC × East/North/South/West with nflverse tricodes. Aliases: `OAK→LV`, `SD→LAC`, `STL/SL/LAR→LA`, `WSH→WAS`, `JAC→JAX`, `ARZ→ARI`, `BLT→BAL`, `CLV→CLE`, `HST→HOU`, plus PFR codes and historical full names.
* **Stadium registry** (`FOOTBALL_STADIUMS`): coordinates, IANA timezone, `roof` ∈ {`dome`, `closed_retractable`, `open_retractable`, `outdoors`} (default configuration; observed roof state is Milestone 27), `surface` ∈ {`grass`, `turf`}, `altitude_m` (Denver ≈ 1609 m), 30 active venues, relocation-era venues, Super Bowl host seasons, and neutral/international venues (London ×3, Munich, Frankfurt, Mexico City, São Paulo, Madrid, Dublin) flagged `is_neutral_site` / `is_international`. `is_neutral_game(season_type, venue_id)` is true for Super Bowls and neutral venues.
* **League priors** (`get_league_priors`): era-based seed points-per-team-game; Milestone 24 replaces them with empirical prior-season league means (passing-era drift).

---

## 5. Elo, Hooks & Schedule Math

### 5.1 Margin-of-Victory, Tie-Aware Elo
`FOOTBALL_ELO_CONFIG` (`base 1500`, `K = 20`, HFA `48`, offseason retention `2/3` ⇒ ~1/3 regression, playoff `K × 1.2`) drives the core `PairwiseEloEngine` (`compute_football_elo_ratings`):

$$\Delta E_{\text{home}} = K_{\text{eff}} \cdot \ln(|\Delta_{\text{pts}}|+1)\cdot\frac{2.2}{0.001\,(E_{\text{winner}}-E_{\text{loser}})+2.2}\cdot (S - P_{\text{home}})$$

* The Elo gap is the **signed winner-perspective** gap (favorites that blow out opponents are damped; upsets are amplified): `compute_winner_perspective_mov_multiplier`.
* Ties use `S = 0.5` with no MOV multiplier; updates are zero-sum.
* Neutral/international/Super Bowl games set HFA to `0` through the `is_neutral_site` column (`prepare_football_elo_games`).
* Franchise aliases are normalized before rating so relocated clubs keep their history.

### 5.2 Starting-QB Hook (`FootballQBEloHook`)
$$\Delta E_{\text{QB}} = \kappa_{\text{QB}}\cdot(\text{QBValue}_{\text{starter}}-\text{QBValue}_{\text{team baseline}})$$
`kappa_qb = 200` by default and the result is clipped to ±250. Starter/baseline values come from any callable returning a finite float or `None` (⇒ `0.0`); `QBValue` is the shrunk rolling EPA+CPOE composite supplied by Milestone 24. A starter→backup downgrade lowers the pre-game Elo gap by exactly `κ · ΔQBValue`.

### 5.3 Venue Hook (`FootballVenueEloHook`)
* **Altitude**: `12 · max(0, home_excess_km − away_excess_km)` Elo where excess is `max(0, altitude − 1000 m)/1000` and the home team owns the venue.
* **Dome**: `+2` when a covered home venue hosts a team whose home venue is open-air.
* **Rest differential**: `3 · clip(home_days_rest − away_days_rest, ±7)`.
* Neutral/international events zero the altitude/dome edges (HFA already `0`) but keep the rest differential.

### 5.4 Weekly-Cadence Features (`core/schedule.py` + `features/schedule_hooks.py`)
Computed per `(team, season)` from the **previous** kickoff only:
* `rest_hours`, `days_rest` = exact kickoff-to-kickoff time (null for a season opener).
* `is_short_week`: `days_rest ≤ 5` (Sunday → Thursday ≈ 4.3 days).
* `is_post_mini_bye`: `9 ≤ days_rest < 13` (e.g., Thursday → Sunday week).
* `is_post_bye`: `days_rest ≥ 13` (Sunday → Sunday with a bye ≈ 14 days).
* `prior_game_was_overtime`: previous game's `overtime` flag.
* NFL hook: `is_thursday_game` / `is_monday_game` (Eastern weekday), `is_thursday_short_week`, `kickoff_local_hour` (venue timezone), `kickoff_local_hour_vs_body_clock` (kickoff wall-clock hour in the team's home timezone, e.g., a West Coast team at 1 PM ET = `10.0`), `is_international_travel`, `travel_km_prev_game`, and `tz_shift_hours`.

---

## 6. Temporal Boundary Rules
1. Sort strictly by `(start_time_utc, event_seq, game_id)`; Thursday → Sunday → Monday ordering is preserved by kickoff instants.
2. Pre-game state only: ratings are recorded before post-game updates; cadence features use the prior appearance and the scheduled kickoff of the current game.
3. Mutating the score, overtime flag, or kickoff of any game at index `≥ t` leaves features for games `< t` bit-for-bit unchanged (covered by `tests/sports/football/test_football_schedule_and_elo.py`).
4. Odds, observed weather, and post-game availability are quarantined from feature tables (Milestones 23, 24, 27, 29).
