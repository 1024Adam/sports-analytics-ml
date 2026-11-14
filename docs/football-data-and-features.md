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
* **Player logs** (weekly player stats + snap share): the nflverse `gsis_id` is the canonical key (`player_id` = digits of the id); `snaps`/`snap_share` come from `snap_counts` via the `players` `pfr_id -> gsis_id` crosswalk and degrade to null when unavailable. Milestone 26 adds `completions`, `pass_air_yards`, `rec_air_yards` (weekly stats) and play-level usage `rz_targets`, `rz_carries`, `gl_carries` (inside the 20 / inside the 5, from pbp `yardline_100`).
* **QB logs** (dropback level, `id` column): `dropbacks`, `pass_attempts`, `qb_epa_total`, `cpoe_total`/`cpoe_plays`, `sacks`, `air_yards_total`, `scrambles`, and `is_starter` (first dropback QB).

* **Snap logs** (`snap_game_logs.parquet`, Milestone 27, from `snap_counts` + the `pfr_id -> gsis_id` crosswalk): one row per `(game_id, team, player_id)` for **every** position (linemen, linebackers, defensive backs included) with `position`, `offense_snaps/offense_pct`, `defense_snaps/defense_pct`. `player_game_logs` only covers stat-sheet players, so position-group availability weights come from here.

* **Micro logs** (`team_pbp_micro_logs.parquet`, `player_pbp_micro_logs.parquet`, Milestone 28): offense-side additive numerators/denominators per `(game_id, team)` and per `(game_id, team, player_id)` built from play-by-play and Next Gen Stats; see section 12.

Uniqueness: `(game_id, team)`, `(game_id, player_id)`, `(game_id, team, player_id)` for snap and player micro logs, `(game_id, team, qb_id)` (`FootballAdapter.validate_team_game_logs` / `validate_participant_schema`).

### 3.3 `team_week_availability.parquet` (as-of snapshots)
One row per `(game_id, team, player_id, source, as_of_time_utc)` with `source` $\in \{\text{injury\_report}, \text{depth\_chart}, \text{inactives}\}`, `report_status`, `practice_status`, `depth_rank`, and `is_inactive`. Stamps are conservative (a later stamp can only hide information):
* Injury reports use `date_modified`, else `kickoff - 24 h`.
* Depth charts use `dt` when present, else `kickoff - 2 days`. All formations are kept (offense, defense, special teams). The 2025+ nflverse layout (`dt`, `team`, `pos_abb`, `pos_rank`, no `season`/`week`) is normalized by assigning each timestamped snapshot to the team's next scheduled game at or after `dt`.
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

---

## 7. Gold Features (`sports/football/features/`, Milestone 24)

`build_football_stage1_features` (and `sports-ml features build --sport football --league nfl`) writes `data/gold/football/stage1_features.parquet`: one row per game (completed **and** upcoming), identifiers, labels (`home_score`, `away_score`, `home_win`, `is_tie`; ties are excluded from classifier targets in Milestone 25), and `STAGE1_FEATURE_COLUMNS` (home-minus-away `diff_*` plus selected `home_*`/`away_*` columns). Boolean features are cast to `Int8`; there are **zero `NaN`/null** feature values.

### 7.1 Play-level filters (Silver `team_game_logs`)
`f_off_*` columns aggregate only scrimmage `pass`/`run` plays with offense win probability in `[0.05, 0.95]` (garbage time removed; unknown WP excluded) and exclude QB kneels and spikes, so neither enters an EPA/play or success-rate denominator. Early downs = downs 1-2; explosive = pass >= 20 yd or run >= 10 yd. Points per drive counts drives whose first scrimmage play is inside the band (TD = 7, FG = 3, else 0). Raw red-zone drives/TD drives and fumbles are unfiltered. Defensive metrics are the opponent's offensive numerators/denominators in the same game.

### 7.2 Team rolling features and cold start (`team.py`)
Metrics (`epa_per_play`, `pass_epa_per_dropback`, `rush_epa_per_rush`, `success_rate`, `early_down_success_rate`, `explosive_play_rate`, `sack_rate`, `turnover_rate`, `points_per_drive`, and offense-only `penalty_yards_per_game`) exist for offense (`off_*`) and defense (`def_*`) with four variants: season-to-date (`_std`), rolling 4/8 games (`_roll4`, `_roll8`, `pregame_exposure_rate`), and EWMA (`_ewma`, alpha 0.30). All use `shift(1)` inside `[team, season]`.

$$\hat r = \frac{n\,r_{\text{curr}} + k\,\pi}{n + k},\quad k = 8\ \text{games},\quad \pi = w\,r_{\text{team, prev season}} + (1-w)\,r_{\text{league, prev season}}$$

* `n` = games played before `t` (windows use `min(n, window)`), so the current-season weight `n/(n+k)` increases monotonically with games played and tends to 1; there is **no hard-coded week cutoff**.
* A franchise without a prior season (expansion/relocation key) or a first season in the data falls back to the previous league mean, then to a per-metric league default.
* **Roster continuity** `c` = share of the team's prior-season offensive usage (snaps, else attempts + carries + targets) held by players on the latest depth-chart snapshot at or before `kickoff - 90 min` (default `0.60`, flagged `continuity_is_imputed`). `w = 0.35 + 0.50 c`, multiplied by `0.85` after a head-coach change (optional `head_coach_changed` input).
* **Opponent adjustment**: a ridge regression `EPA/play = mu + off[team] + def[opp]` (weights = plays) fit on the season's games from *earlier weeks* only, shrunk (lambda = 8) toward `w x` the prior-season ratings (`adj_off_epa`, `adj_def_epa`, `adj_epa_net`).
* **Luck/regression**: `turnover_diff_pg` and `turnover_regression_signal` (shrunk minus raw, pseudo-count 16 games), `fumble_luck_pg` (`0.5 x fumbles - fumbles lost` per game), `close_game_win_pct` (one-score games, shrunk to 0.5), `pythag_win_pct` (exponent 2.37, prior-season blend), `pythag_luck` (win% minus Pythagorean, shrunk), and `rz_td_rate` (red-zone TD%, 20-drive pseudo-count toward 0.57).
* Upcoming games reuse the last completed state; games beyond the next are projections from it.

### 7.3 Starting QB (`qb.py`)
Player-level, team-agnostic history: decayed (`0.96`/game) career sums from games before `t`, shrunk toward a replacement prior (`k = 250` dropbacks): `qb_epa_per_dropback`, `qb_cpoe`, `qb_sack_rate`, `qb_air_yards_per_attempt`, `qb_rush_epa`, `qb_games_with_team`, `qb_career_games`, `qb_is_rookie_or_debut`, `qb_is_backup` (starter is not the team's most-dropback QB over its last 17 games), and `qb_value = EPA/dropback + 0.01 CPOE`. Debut/rookie priors use draft capital: `prior_EPA = -0.10 + 0.15 e^{-(pick-1)/40}` and `prior_CPOE = -2.0 + 3.0 e^{-(pick-1)/40}` (`draft_picks`). `team_qb_baseline_value` is an EWMA (alpha 0.15) of the team's previous starters' values; `qb_value_over_baseline` and the `(game, team) -> value` maps feed `FootballQBEloHook`. Unknown starters of unplayed games project the incumbent (`qb_is_imputed`).

### 7.4 Schedule context and Elo (`schedule_context.py`)
Home/away `days_rest`, `is_short_week`, `is_post_bye`, `is_post_mini_bye`, `prior_game_was_overtime`, `travel_km`, `timezone_shift`, `consecutive_road_games`, `kickoff_body_clock_hour`; game-level `rest_diff`, `is_thursday_game`, `is_monday_game`, `is_international`, `is_neutral_site`, `is_division_game`, `is_conference_game`, `is_playoff`, `playoff_round` (REG 0, WC 1, DIV 2, CON 3, SB 4). A missing rest (season opener) is imputed to 7 days with `*_days_rest_is_imputed`. `elo_home_pre`, `elo_away_pre`, `diff_elo_pre`, `elo_home_win_prob` come from the plain Elo; `qb_elo_diff_pre` / `qb_elo_home_win_prob` add the starting-QB hook and the venue hook (altitude, dome, rest differential).

### 7.5 Hard odds & observed-outcome firewall (`pipeline.py`)
`is_forbidden_feature_column` rejects names starting with `market_` or `obs_weather_` or containing `odds`, `spread`, or `closing`. `STAGE1_FEATURE_COLUMNS` / `STAGE2_FEATURE_COLUMNS` are filtered and asserted at import time, and the builder strips such columns from its inputs before any feature code runs, so mutating `market_*` / `obs_weather_*` Silver columns changes `0.0` of the output (`tests/sports/football/test_football_odds_firewall.py`).

---

## 8. Stage 1 Model (`sports/football/models/`, Milestone 25)

`Stage1FootballGameModel` consumes the Gold matrix (`STAGE1_FEATURE_COLUMNS` present in the frame, firewall-checked) and returns, per game, `home_win_prob`, `away_win_prob = 1 - home_win_prob`, `draw_prob = 0.0`, `raw_home_win_prob`, `expected_home_score`, `expected_away_score`, `expected_margin`, and `expected_total` (upcoming games included).

* **Expected scores**: two shallow (depth 3, 4 leaves), heavily regularized Tweedie LightGBM regressors with a log link anchored on the previous season's mean points per team-game (`init_score`; falls back to `get_league_priors`), so `mu = anchor * exp(raw)`. Monotone constraints: more QB value / EPA / Elo edge never lowers the home score (mirrored for the away regressor). Ties stay in the score targets.
* **Win-probability ensemble** (logit-space blend, default weights `0.40 / 0.30 / 0.30`): QB-adjusted Elo (`qb_elo_home_win_prob`), the score-implied probability $\Phi(\text{margin}/\sigma_{\text{margin}})$ with $\sigma_{\text{margin}}$ estimated on the held-out calibration block and shrunk toward 13.5 (clipped to 11-16), and a regularized logistic (`C = 0.05`) + shallow-GBDT classifier. **Ties are dropped from classifier fit and calibration** (they remain `0.5` in Elo).
* **Calibration**: `core.calibration.BinaryProbabilityCalibrator` (`"platt"` default or `"temperature"`, never isotonic) fit on pooled predictions from the latest `calibration_seasons = 2` seasons that the GBDT block did **not** train on. It falls back to the identity map (`is_identity`, `fallback_reason`) when fewer than `min_calibration_samples = 400` non-tie games are available, only one class is present, or the Platt slope is not positive.
* **Neutral sites** (`is_neutral_site == 1`) are symmetric under home/away relabeling: predictions average the original and swapped orientation (`swap_home_away_features` swaps `home_*`/`away_*`, negates `diff_*`, flips home-win probabilities).
* **Walk-forward backtest** (`run_football_stage1_walk_forward_backtest`): expanding, week-aligned folds that test on whole future seasons (`test_unit="season"`) or week blocks (`"week_block"`). Each fold satisfies `gbdt_train_max_time < calibrator_min_time <= calibrator_max_time < test_min_time` (`FoldBoundaryAudit`). Reported: Log Loss, Brier, ECE, accuracy (ties excluded), score MAE/RMSE (home, away, margin, total) against a league-mean baseline, coin-flip, empirical home-win-rate prior (also per era: 1999-2010, 2011-2019, 2020+), plain and QB-adjusted Elo baselines, playoff-only and Weeks 1-4 slices, and **season-bootstrap 95% intervals** of the log-loss difference versus the home prior and QB-adjusted Elo.
* **Bundle** (`training.py`): `train_football_stage1_bundle` fits on all completed games and saves `artifacts/football/<version>/` and `latest/` (`stage1_model.pkl`, `metadata.json` with the SHA-256 schema fingerprint) via `ModelRegistry`; `load_football_stage1_bundle` restores it.

Reference result on real nflverse data (2012-2024 Silver, 8 test seasons 2017-2024, `min_train_seasons=5`): log loss `0.624` vs QB-Elo `0.631` (95% CI of the difference `[-0.011, -0.001]`), plain Elo `0.635`, home prior `0.690`; ECE `0.018`; margin MAE `10.19` vs `11.22` (zero-margin baseline). Expected totals are not better than the league-mean baseline (`11.12` vs `11.07` MAE) because the Stage 1 features are mostly home-minus-away differences; level features (pace, PROE, weather) arrive in Milestones 27-28.

## 9. Stage 2 Player Projections (`features/player.py`, `models/stage2_player.py`, Milestone 26)

**Gold rows** (`build_football_stage2_features`, persisted by `build_and_persist_football_stage2_features` to `data/gold/football/stage2_features.parquet`): QB, RB, WR and TE (FB/HB map to RB) per `(game_id, team, player_id)`. A chronological loop emits each game's rows *before* updating decayed (0.93/game) per-player state, so game `T` features depend only on games `< T`. Completed games emit actual participants with `actual_*` labels; upcoming games emit the team's recent contributors (QB 2, RB 4, WR 5, TE 3) plus any resolver-named starter, with null labels.

* **Features** (`STAGE2_PLAYER_FEATURE_COLUMNS`, firewall-checked, null/NaN-free): `p_active` (+`p_active_is_imputed`; injury/inactive snapshots stamped `<= min(as_of, kickoff - 90 min)`, default 1.0), debut / draft-capital factor, shrunk shares (`snap_share`, `target_share`, `air_yards_share`, `wopr`, `route_participation` proxy = snap share, `rz_target_share`, `carry_share`, `gl_carry_share`, `rz_carry_share`, `pass_att_share`), shrunk rates (completion, yards per attempt/carry/target, TD per attempt/carry/target, INT, QB rushing), opposing-defense indices vs. position (pass/rush/receiving yards allowed, 1.0 = league), and team context (Elo win probability, pace, pass rate as a PROE proxy). Shrinkage priors come from `POSITION_PRIORS`/`STAGE2_SHRINK_K` with a draft-capital share bonus (`DRAFT_SHARE_BONUS`).
* **Model** (`Stage2FootballPlayerModel`, a structural model rather than a GBDT): (1) team opportunity - non-negative ridge for team pass and rush attempts from expected margin, expected total, pace and pass rate, with a game-script floor so a higher expected margin always shifts volume from the pass to the run; (2) player lines = opportunity x active-weighted shares x shrunk rates x defense index; (3) `reconcile_football_stage2_projections` (`gamma = 0.5`) conserves opportunity (QB attempts sum to team pass attempts; targets and RB carries never exceed team totals; an inactive player's volume is redistributed, never created) and soft-scales yards and TDs toward Stage 1 expected-points totals; (4) `prob_anytime_td = p_active * (1 - exp(-lambda))` and zero-inflated gamma thresholds (`prob_250plus_pass_yds`, `prob_300plus_pass_yds`, `prob_2plus_pass_td`, `prob_75plus_rush_yds`, `prob_100plus_rush_yds`, `prob_5plus_rec`, `prob_75plus_rec_yds`, `prob_100plus_rec_yds`), with `prob_100plus <= prob_75plus` guaranteed.
* **Training**: Stage 2 conditions on *in-sample* Stage 1 expected scores (a mild, documented optimism); `evaluate_football_stage2_holdout` fits on earlier seasons and scores the newest seasons against out-of-sample Stage 1 scores. Historical rows contain only players who actually played (a selection caveat the holdout MAE inherits); `route_participation` and PROE are proxies until snap/route data is wired in.

## 10. Unified Predictor and CLI (`models/predictor.py`, `reporting.py`, Milestone 26)

`FootballMatchupPredictor.from_artifacts(artifacts/football/latest)` exposes `predict_matchup(season, week, away, home)`, `predict_week`, `predict_date`, and `predict_team_next_games`. Gold is rebuilt from Silver with the 2-tier resolver's starter injected for not-yet-final games; Stage 2 rows for the requested games are rebuilt with the target games' realized participants withheld (so retrospective predictions never see who actually played), and each QB row's `p_active` is scaled by the resolver's `P(starts)`.

| Warning | Meaning |
| --- | --- |
| `CONFIRMED_QB` / `PROBABLE_QB` / `QB_UNCERTAIN` | Tier 1 (confirmed or depth chart minus inactives, or the game is final) / Tier 2 starter with `P(starts) >= 0.80` / lower-confidence or incumbent-imputed starter |
| `BACKUP_QB`, `DEBUT_OR_NEW_QB_PRIOR` | A side starts a non-incumbent QB / a rookie or QB with no games for the team |
| `EARLY_SEASON_SHRINKAGE` | Either team has played fewer than 4 games (prior-season shrinkage dominates) |
| `NEUTRAL_OR_INTERNATIONAL_SITE` | Neutral-site or international game (symmetric home/away prediction) |
| `KEY_OFFENSIVE_PLAYER_OUT` | A high-usage QB/RB/WR/TE has `p_active < 0.5` |
| `STAGE1_ONLY_NO_PLAYER_DATA` | `--include-players` was requested but no Stage 2 model/rows are available |

Ledgers: `data/predictions/football_predictions.parquet` (one row per prediction, deduplicated by `game_id`, latest wins, when scored) and `football_player_predictions.parquet`. `score_football_prediction_ledger` reports log loss/Brier/accuracy against home-rate and coin-flip baselines (plus the market-edge report when closing odds exist) and player projection MAE. Predictions made for already-final games are retrospective and are scored the same way, so use them for smoke tests rather than live evaluation.

CLI: `features build`, `train [--include-stage2]`, `backtest [--stage 1|2|all] [--splits N]`, `predict [--season --week | --date | --team --games] [--matchup AWAY@HOME] [--include-players] [--offline]`, and `score-predictions` all accept `--sport football` (`--league nfl`). Output shows `P(Away)`/`P(Home)`, projected score, a QB matchup card, and position-grouped projection tables with `P(Anytime TD)`.

## 11. Weather, Roof/Surface & Availability Features (Milestone 27)

**Forecast weather** (`ingestion/weather.py`, cached under `data/raw/football/weather/`). Historical games use the Open-Meteo *previous runs* archive (`temperature_2m`, `wind_speed_10m`, `precipitation_probability` with the `_previous_day1` suffix, i.e. the forecast issued about 24 h before the valid time, `forecast_lead_hours = 24`); upcoming games (within 14 days) use the live forecast with `forecast_lead_hours = hours until the valid time`. `ingest --sport football` syncs the outdoor/open-roof venue windows (failures only warn; `--offline` skips the sync and Gold builds read the cache only). Hourly rows are averaged over the four hours starting at the kickoff hour (precipitation: maximum).

Archive coverage is the binding constraint: temperature exists from 2021, but wind and precipitation probability only from 2024, so most historical seasons use the observed proxy below. `build_game_weather` returns one row per game:

| Case | `weather_temp_f / wind_mph / precip_prob` | Flags |
| --- | --- | --- |
| Dome / closed roof (stadium registry default roof, never the observed game-day roof) | 70 F / 0 mph / 0 | `weather_is_indoor = 1` |
| Season where >= 90% of outdoor games (up to the 14-day forecast horizon) have a complete forecast | forecast; neutral 55 F / 7 mph / 0.15 if a game is missing | `weather_is_imputed = 1` for the missing game |
| Any other season | **observed** nflverse `temp`/`wind` for *every* outdoor game of that season; precipitation neutral | `weather_is_observed_proxy = 1`, `weather_precip_is_imputed = 1` |

Forecasts and observations are therefore never mixed inside a season, raw `forecast_*` columns stay null where no forecast exists, and `obs_weather_*` remain quarantined (Gold builders still strip them; only `build_game_weather` reads them, for the flagged proxy).

**Environment features** (`features/environment.py`; `ENVIRONMENT_FEATURE_COLUMNS`): `wind_penalty = ((wind - 15) / 10)^1.5` above 15 mph, `cold_penalty = (40 - temp) / 10` below 40 F, `is_precipitation_risk` (prob >= 0.5), all scaled by `roof_exposure` (1.0 outdoors, 0.5 open-retractable, 0 covered); `home_/away_warm_team_in_cold` (+ `diff_`) for dome or warm-climate (latitude <= 35.3) teams playing outdoors in the cold away from their own venue; `env_passing_penalty`, `env_kicking_penalty`, and `env_total_shift` (expected-points prior: `-(1.2 wind + 0.3 cold + 0.8 precip)`); surface/altitude interactions `cold_penalty_x_turf`, `precip_risk_x_grass`, `altitude_kicking_boost`. Stage 1 applies same-sign monotone constraints (`LEVEL_MONOTONE_CONSTRAINTS`: worse weather never raises either team's expected score) and Stage 2 multiplies yards per attempt/target by `1 - 0.05 * env_passing_penalty` (floor 0.75) through the `env_passing_penalty` feature. Without a weather frame, outdoor games are neutral and flagged `weather_is_imputed = 1`.

**Availability features** (`features/availability.py`; `AVAILABILITY_FEATURE_COLUMNS`, `home_*`/`away_*`/`diff_*`): for each team-game the weight of every player is his mean snap share over the team's last 6 released games; `ol_`, `pass_rush_`, `secondary_`, `skill_availability_index = sum(w * p_play) / sum(w)` per position group (OL: T/G/C; pass rush: DE/DT/DL/NT/EDGE/OLB; secondary: CB/S/SS/FS/DB; skill: RB/FB/WR/TE), `starter_snap_share_missing` (weight-averaged `1 - p_play` over players with weight >= 0.5), and `qb_p_starts` (probability the depth-chart QB starts, resolver-style). `availability_is_imputed` marks team-games without snapshots or snap history.

* `p_play = 0` for official inactives stamped by the cutoff, otherwise a **historical start probability** for the latest injury designation: `(played + 10 * prior) / (total + 10)` with `prior = status_play_probability(...)` (Out 0, Doubtful 0.05, Questionable 0.60, practice levels 0.85-1.0). `played` is "has snaps in the snap log"; the counts include only games released before the prediction time. No designation means the player is assumed to play.
* **As-of rules**: snapshots stamped after `min(kickoff - 90 min, as_of_time_utc)` are ignored; a game's outcomes (snap logs, start-probability counts) are released 4 h after its kickoff and never after `min(kickoff, as_of_time_utc)`.

**Backtest impact (real nflverse data, 2012-2026 Silver, 10 test seasons 2017-2026, `min_train_seasons = 5`)**:

| Stage 1 feature set | Log loss | Brier | ECE | Margin MAE | Total MAE |
| --- | --- | --- | --- | --- | --- |
| Before (no Milestone 27 columns) | 0.6240 | 0.2171 | 0.0233 | 10.144 | 11.150 |
| + environment only | 0.6243 | 0.2174 | 0.0243 | 10.161 | 11.126 |
| + availability only | 0.6235 | 0.2169 | 0.0267 | 10.147 | 11.166 |
| After (both) | 0.6241 | 0.2173 | 0.0207 | 10.179 | 11.135 |

Season-bootstrap 95% intervals of the after-minus-before differences all include zero: log loss `+0.0000 [-0.0017, +0.0019]`, margin MAE `+0.032 [-0.006, +0.082]`, total MAE `-0.018 [-0.064, +0.034]`. Stage 1 therefore does not regress; the features mainly matter for Stage 2 passing environments and as inputs for the in-season signals (forecasts and live availability) that the historical proxy cannot show.

## 12. Play-by-Play Micro-Features (Milestone 28)

**Silver micro logs** (`ingestion/pbp_micro.py`, `ingestion/ngs.py`; written by `ingest --sport football`, with Next Gen Stats fetched per season). Every column is an additive numerator or denominator, so rolling rates are exposure-weighted and the defensive view of a metric is the opponent's row.

* `epa_total` / `plays` use exactly the `team_game_logs` scrimmage definition and reconcile to `off_epa_total` / `off_plays` (tested to `1e-6`). All other metrics exclude QB kneels and spikes.
* **Neutral situation** (`NeutralSituationConfig`): win probability in `[0.20, 0.80]`, `|score margin| <= 14`, more than 120 s left in the half, regulation, and down `<= 3`. It feeds `neutral_seconds` / `neutral_seconds_plays` (seconds from a snap to the next snap of the same drive, kept only when in `(0, 60]` s), `proe_num` / `proe_den` (`pass - xpass`, with nflfastR's `xpass`), and `neutral_epa`.
* Pass game: `cpoe_*`, `air_yards_*`, `adot_*` (targeted plays), `yac_oe_*` (`yards_after_catch - xyac_mean_yardage`), `pressure_num`, `sacks`. nflverse play-by-play has **no charted pressure or time-to-throw**, so `pressure_rate` is a proxy, `(sack or qb_hit) / dropbacks`; time to throw comes only from NGS. Situational: red-zone and third-down EPA, `explosive_pass`, `run_stuffs`, penalty-adjusted EPA (`nopen_*`), two-minute EPA when tied or trailing, and fourth-down go rate in the go zone (`ydstogo <= 4`, own side of midfield plus 15, win probability in the garbage-time band).
* **NGS** (`ngs_*`; passing: time to throw and CPOE weighted by attempts, rushing: rush yards over expected, receiving: separation and YAC over expected weighted by targets/receptions): weekly rows from 2016 on (`week == 0` season rows are ignored). `has_ngs = 0` marks team-games without NGS rows, so seasons before 2016 carry zeros and resolve to the league prior.
* **Player micro logs**: receiver targets, air yards, receptions, receiving yards, YAC over expected, rush attempts/yards, plus NGS rushing/receiving totals (`ngs_ryoe_sum`, `ngs_rush_attempts`, `ngs_separation_sum`, `ngs_targets`).

**Rolling team features** (`features/micro.py`): each metric is the exposure-weighted rate over the team's previous 8 games of the season (`shift(1)` inside `[team, season]`), shrunk with `bayesian_shrinkage` (`k = 8` games that carry the metric's exposure) toward `0.5 * last season's team rate + 0.5 * last season's league rate` (league default for the first season of data). `micro_is_imputed` / `ngs_is_imputed` flag team-games without any micro / NGS history; imputed rows are exact league priors, never NaN. Per-team values are offense-only or `off - def` / `def - off` edges so higher is better (pace stays raw seconds per play; `fourth_down_go_over_expected` subtracts last season's league go rate).

**Stage 1 ships five matchup columns** (`STAGE1_MICRO_FEATURE_COLUMNS`): `diff_proe`, `diff_pressure_rate_edge`, `diff_neutral_pace`, `diff_red_zone_epa`, `diff_cpoe_vs_coverage` (net passing CPOE minus CPOE allowed), plus `home_/away_micro_is_imputed`. The wider set (`MICRO_FEATURE_COLUMNS`: 17 diffs including NGS time to throw, rush yards over expected, separation, third-down, penalty-adjusted and two-minute EPA, plus `pace_total_neutral` / `proe_total`) is built by `build_micro_team_features` + `attach_micro_features` but kept out of Gold because it did not improve the backtest:

| Real nflverse data, 2012-2026, 10 test seasons, `min_train_seasons = 5` | Log loss | Brier | Margin MAE | Total MAE |
| --- | --- | --- | --- | --- |
| Before (Milestone 27 features) | 0.6241 | 0.2173 | 10.179 | 11.135 |
| + five shipped micro diffs | 0.6240 | 0.2172 | 10.159 | 11.112 |
| + all 17 micro diffs + totals | 0.6246 | 0.2175 | 10.160 | 11.116 |
| + play-by-play-only subset (no NGS) | 0.6252 | 0.2177 | 10.180 | 11.108 |

Season-bootstrap 95% intervals of the shipped-minus-before difference: log loss `-0.0002 [-0.0006, +0.0003]`, margin MAE `-0.016 [-0.047, +0.015]`, total MAE `-0.023 [-0.043, -0.003]` (a small gain in expected totals; no regression anywhere).

**Stage 2** (`features/player.py`): `yprr` (receiving yards per route run, a **proxy**: routes = offensive snaps x the team's dropback share, shrunk toward `YPRR_PRIORS` with `k = 80`), `rush_yards_over_expected_per_att` (NGS rush yards over expected per carry, shrunk toward 0 with `k = 40`, RB/QB only), and `ngs_is_imputed` join the existing `air_yards_share`/`target_share`. All three are updated only after a game's rows are emitted. `Stage2FootballPlayerModel` adds `0.25 * rush_yards_over_expected_per_att` to yards per carry and scales yards per target by `1 + 0.15 * (yprr / prior - 1)` (clipped to 0.85-1.15). Holdout MAE (last three seasons with player rows) before to after: RB rush yards `21.064 -> 21.014`, WR receiving yards `23.673 -> 23.593`, all other stats unchanged. Real separation and routes charts are not available, so `separation` only appears in the experimental team columns.

## 13. Closing-Odds Benchmark, De-Vigging & ATS / Total Evaluation (Milestone 29)

**Ingestion** (`ingestion/odds.py`, `FootballOddsIngestor`; runs at the end of `ingest --sport football`). Three sources fill the nullable `market_*` columns of Silver `games.parquet` (`market_home_decimal_odds`, `market_away_decimal_odds`, `market_prob_home`, `market_prob_away`, `market_spread_home`, `market_total_points`):

1. The quarantined nflverse schedule columns always win (a value they provide is never overwritten).
2. An optional local archive `data/raw/odds/nfl_historical_odds.csv.gz` (or `.csv`) fills what the schedule lacks. Columns: `nflverse_game_id` or `game_date,home_team,away_team` (full names and legacy codes are normalized); `home_moneyline`/`away_moneyline` (American) or `home_decimal_odds`/`away_decimal_odds`; `spread_home` (betting convention, negative when the home team is favored) or nflverse-style `spread_line` (positive when home is favored); `total_points` or `total_line`. Rows match by id, else by teams with a kickoff date within one day.
3. An optional `THE_ODDS_API_KEY` snapshot (`h2h,spreads,totals`, averaged across bookmakers) fills only games that are not final; it is skipped silently without a key, with `--offline`, or on any API failure.

Moneyline pairs are **power de-vigged** (`core.market.devig_probabilities`), so `market_prob_home` and `market_prob_away` lie in `(0, 1)` and sum to 1 within `1e-9`. The two-sided quote (odds and probabilities) is taken whole from a single source or not at all.

**Firewall.** Market data is evaluation-only. `STAGE1_FEATURE_COLUMNS` / `STAGE2_FEATURE_COLUMNS` contain no `market_*`, `*odds*`, `*spread*`, `*closing*`, `*moneyline*` names and none starting with `spread`/`total`; every feature builder strips `market_*`/`obs_weather_*` from its inputs. Tests mutate every market column and assert bit-for-bit identical Gold features, probabilities, and expected scores while the market reports change.

**Evaluators** (`core/market.py`, additive; wrapped by `core.evaluation.evaluate_line_markets`):

* `evaluate_spread_betting(model_margin, closing_spread_home, actual_margin)`: edge = `model_margin + spread_home` (positive favors the home side). The home side covers when `actual_margin + spread_home > 0`, the away side when `< 0`, and a whole-point line pushes at exactly 0 (half-point lines never push). Hit rate excludes pushes; ROI is per unit staked at `-110` (win `+100/110`, loss `-1`, push refunded), and the break-even win rate is `110/210 = 52.38%`.
* `evaluate_total_betting(model_total, closing_total, actual_total)`: edge = `model_total - closing_total` (positive bets the over); pushes at the number.
* Buckets by disagreement `tau in {1, 2, 3}` points (a bet is placed when `|edge| >= tau`) report `W-L-P`, hit rate, ROI, mean `|edge|`, and **key-number** handling: bets whose closing spread is 3, 6, 7 or 10 in absolute value (`FOOTBALL_KEY_NUMBERS`) and how many of them pushed. The report also states whole- vs half-point line counts and the model's versus the closing line's MAE.
* **Closing-line value** (`closing_line_value`): a home bet taken at `+3` that closes at `+1` earned `+2` points (an over taken at 44 that closes at 46 earned `+2`; away/under flip the sign). CLV needs a line at bet time, so it appears only where one exists: the live ledger records `market_spread_home_at_pred`, `market_total_at_pred`, and `market_prob_home_at_pred`, and `score-predictions` compares them with the final closing line. Backtests have no bet-time line and omit CLV; predictions made for already-final games record the closing line itself (CLV 0).

**CLI.** `backtest --sport football` (Stage 1) and `score-predictions --sport football` print, when odds exist: *De-Vigged Market Closing Line* (model vs market on the same games), *Fundamental vs. Market Disagreement* slices, *Against the Spread (ATS)*, and *Over / Under vs. Closing Total*; without odds they print "market benchmark skipped".

Reference run (real nflverse closing lines, 2012-2026 Silver, last six seasons held out): market log loss `0.6128` / Brier `0.2126` versus Stage 1 `0.6263` / `0.2182`; ATS hit rate `48.4%` (1+ pt), `50.8%` (2+), `51.5%` (3+), all below the 52.38% break-even (ROI `-7.4%`, `-3.0%`, `-1.6%`); over/under `48-49%`. Model margin MAE `10.03` versus closing-line `9.75`; total MAE `11.04` versus `10.33`. In other words Stage 1 does not beat the closing line; the closing market remains the benchmark to close in on.
