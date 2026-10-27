# Data Contracts & Feature Engineering Specification

> **Related Documents**: [System Architecture Blueprint](./architecture.md) | [Implementation Roadmap](./roadmap.md)

## 1. Hybrid NHL Data Source Strategy

The hockey plugin combines two complementary, publicly accessible data sources so the pipeline gets both **official schedule/roster/boxscore structure** and **research-grade expected goals ($xG$) & shot-quality metrics**:

### Source A: Official NHL Web API (`https://api-web.nhle.com/v1/`)
* **Role**: Canonical source of truth for game schedules, start timestamps (UTC), venue metadata, active game rosters, starting goalies, official boxscore counting stats, and daily upcoming matchups.
* **Key Endpoints**:
  * `GET /v1/schedule/{YYYY-MM-DD}` or `/v1/club-schedule-season/{TEAM}/{SEASON}`: Game IDs (`2024020001`), UTC start times, home/away teams, venues, game states (`FUT`, `LIVE`, `FINAL`, `OFF`).
  * `GET /v1/gamecenter/{GAME_ID}/boxscore`: Official skater stats (`goals`, `assists`, `points`, `sog`, `toi`, `powerPlayGoals`, `powerPlayToi`, `shifts`, `pim`, `hits`, `blockedShots`), goalie stats (`shotsAgainst`, `saves`, `goalsAgainst`, `toi`, `starter` flag), and period-by-period scores.
  * `GET /v1/roster/{TEAM}/current` (or season roster): Active player metadata (`player_id`, position `C`/`L`/`R`/`D`/`G`, shoots/catches, birth date).
* **ID Convention**: NHL official 10-digit `game_id` (e.g., `2023020105` where `2023` = season start year, `02` = regular season, `0105` = game number) and 7-digit `player_id` (e.g., `8478402`).

### Source B: MoneyPuck Historical & Season CSV Datasets (`https://moneypuck.com/moneypuck/playerData/`)
* **Role**: High-signal advanced analytics per game and situation (`5on5`, `5on4` PP, `4on5` PK, `all`), including flurry-adjusted Expected Goals ($xGF$, $xGA$), Corsi ($CF$, $CA$), Fenwick ($FF$, $FA$), High-Danger shot attempts, and Goalie Goals Saved Above Expected ($GSAx$).
* **Reconciliation Key**: MoneyPuck uses standard NHL 10-digit `gameId` and NHL 7-digit `playerId`, plus 3-letter team tricode (`BOS`, `TOR`, etc., with historical aliases like `ARI` $\rightarrow$ `UTA`, `S.J` $\rightarrow$ `SJS`, `L.A` $\rightarrow$ `LAK`, `N.J` $\rightarrow$ `NJD`, `T.B` $\rightarrow$ `TBL` normalized in `reconciler.py`).
* **Graceful Fallback**: If MoneyPuck data has not yet updated for a game completed late the night before, the reconciler falls back to NHL API shot/goal/TOI metrics and marks `has_moneypuck_xg = False` without breaking daily inference.

### 1.1 Current-Season-to-Date & Incremental Sync Strategy
To predict tonight's games using all games played in the current season up through yesterday, the ingestion layer distinguishes between **immutable historical data** and **mutable current-season-to-date data**:

1. **Auto-Detection of Active Season**: Calling `sports-ml ingest --sport hockey --current-season` queries `GET /v1/schedule/now` (or derives the active NHL season ID from the current date, e.g., October 2026 $\rightarrow$ `20262027`).
2. **Smart Delta & Compressed Raw-Cache Policy (`core/cache.py`)**:
   * **Compress-on-Ingest (`.csv.gz` & `.json.gz`)**: All raw HTTP CSV and JSON payloads under `data/raw/` are written as Gzip-compressed files (`.csv.gz` and `.json.gz`, with uncompressed `.meta.json` ETag/Last-Modified sidecars). Polars natively streams `.csv.gz` directly in Rust via `pl.read_csv()`, and all readers transparently fall back to legacy uncompressed `.csv` / `.json` files if present on disk.
   * **Completed Games (`gameState in {"FINAL", "OFF"}`)**: Once a game reaches `FINAL`/`OFF` and its boxscore is fetched, its raw JSON is cached permanently on disk (`data/raw/nhl_api/boxscores/{game_id}.json.gz`) and never re-downloaded.
   * **In-Progress / Upcoming Games (`gameState in {"FUT", "PRE", "LIVE"}`)**: Never cached permanently. When you run `--current-season` today, any game from earlier in the week that was previously `FUT` is re-queried; if it is now `FINAL`, its completed boxscore is ingested and appended to the Silver tables.
   * **MoneyPuck Shared `teams_all.csv.gz` & Season Skater/Goalie `.csv.gz` Snapshots**: Because MoneyPuck publishes cumulative historical team game-by-game logs in a single upstream `all_teams.csv` feed, `MoneyPuckClient` caches a single shared `data/raw/moneypuck/teams_all.csv.gz` (`~18 MB` instead of duplicating `122 MB` per season) and filters by `season` in Polars. Season-specific skater and goalie archives are extracted from upstream `.zip` responses and persisted as `skaters_{YYYY}.csv.gz` and `goalies_{YYYY}.csv.gz`, refreshed via `ETag` / `Last-Modified` and TTL for the active season.
   * **Current Active Rosters & 2-Tier Pre-Game Lineup/Goalie Resolution (`lineups.py`)**:
     Because the Official NHL Web API only posts the dressed 20-man game lineup ~30 minutes before puck drop, live matchup predictions resolve tonight's starting goalies, dressed skaters, and injuries using a **2-Tier Live Waterfall**:
     * **Tier 1 — Official NHL Pre-Game / Warmup Lineup (`api-web.nhle.com`)**: Queries `/v1/gamecenter/{GAME_ID}/boxscore` (and `/landing`) plus `/v1/roster/{TEAM}/current`. If official pre-game lineups and starting goalies are already posted (typically ~30 minutes before puck drop), Tier 1 is used directly (`CONFIRMED - NHL Official`).
     * **Tier 2 — Daytime Morning-Skate Lineups, Injuries & Goalie Feed (`DailyFaceoff` + NHL Active Roster)**: Earlier on game day (before official warmup lineups are posted to the NHL API), fetches the morning-skate starting goalie status (`Confirmed` / `Expected`), projected forward lines / defense pairings, and injury list, cross-referenced against `/v1/roster/{TEAM}/current` player IDs so injured/scratched players are excluded from Stage 1 TOI-weighted roster strength and Stage 2 player stat projections.
3. **Upsert into Silver Lakehouse**: Reconciled records are merged (`upsert` on `game_id` / `(game_id, player_id)`) into the Silver Parquet tables, immediately updating rolling windows, Elo ratings, and player per-60 rates through the most recently completed game.

---

## 2. Canonical Polars Parquet Schemas (Silver Layer)

All Silver tables are stored as partitioned Parquet files under `data/silver/hockey/` and strictly sorted by `(game_date, start_time_utc, game_id)`.

### 2.1 `games.parquet` (1 row per game)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `sport` | `pl.Utf8` | Always `"hockey"` for NHL plugin |
| `season` | `pl.Int32` | e.g., `20242025` |
| `game_type` | `pl.Int8` | `2` = Regular Season, `3` = Playoffs (filtered cleanly) |
| `game_id` | `pl.Int64` | Official 10-digit NHL game ID |
| `game_date` | `pl.Date` | Local schedule date of game |
| `start_time_utc` | `pl.Datetime("us", "UTC")` | Exact puck-drop UTC timestamp (used for temporal ordering) |
| `home_team` | `pl.Utf8` | Canonical 3-letter team code (e.g., `"BOS"`) |
| `away_team` | `pl.Utf8` | Canonical 3-letter team code (e.g., `"TOR"`) |
| `venue_timezone` | `pl.Utf8` | IANA timezone of venue |
| `game_state` | `pl.Utf8` | `"FINAL"` for completed historical games; `"FUT"` for upcoming |
| `finish_period_type` | `pl.Utf8` | `"REG"`, `"OT"`, or `"SO"` (`null` if upcoming) |
| `home_goals` | `pl.Int16` | Official final home goals (includes SO winner goal) |
| `away_goals` | `pl.Int16` | Official final away goals |
| `home_reg_goals` | `pl.Int16` | Regulation + OT goals (excludes shootout tiebreaker goal) |
| `away_reg_goals` | `pl.Int16` | Regulation + OT goals |
| `home_win` | `pl.Int8` | Target $y \in \{0, 1\}$ (`1` if `home_goals > away_goals`) |

### 2.2 `team_game_logs.parquet` (2 rows per game — 1 Home, 1 Away)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `game_id` | `pl.Int64` | Joins to `games.parquet` |
| `game_date` | `pl.Date` | Game date |
| `start_time_utc` | `pl.Datetime("us", "UTC")` | Temporal sort key |
| `season` | `pl.Int32` | Season identifier |
| `team` | `pl.Utf8` | Team tricode |
| `opponent` | `pl.Utf8` | Opponent tricode |
| `is_home` | `pl.Boolean` | `True` for home team, `False` for away |
| `goals_for`, `goals_against` | `pl.Int16` | Actual goals in regulation + OT |
| `sog_for`, `sog_against` | `pl.Int16` | Shots on goal |
| `xgf_5v5`, `xga_5v5` | `pl.Float64` | 5-on-5 Expected Goals For and Against |
| `cf_5v5`, `ca_5v5` | `pl.Int16` | 5-on-5 Corsi (all shot attempts) For and Against |
| `ff_5v5`, `fa_5v5` | `pl.Int16` | 5-on-5 Fenwick (unblocked shot attempts) For and Against |
| `hd_xgf_5v5`, `hd_xga_5v5` | `pl.Float64` | 5-on-5 High-Danger Expected Goals For and Against |
| `toi_5v5_minutes` | `pl.Float64` | Total 5-on-5 time on ice in minutes |
| `pp_goals`, `pp_opportunities`, `pp_xgf`, `pp_toi_min` | `pl.Float64` | Power-play offense metrics (`5v4`) |
| `pk_goals_against`, `pk_times_shorthanded`, `pk_xga`, `pk_toi_min` | `pl.Float64` | Penalty-kill defense metrics (`4v5`) |
| `has_moneypuck_xg` | `pl.Boolean` | `True` when MoneyPuck situational $xG$ was joined; `False` when using NHL API boxscore fallback |

### 2.3 `skater_game_logs.parquet` (1 row per active skater per game)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `game_id`, `game_date`, `start_time_utc`, `season` | — | Temporal & game keys |
| `player_id` | `pl.Int64` | NHL 7-digit player ID |
| `player_name` | `pl.Utf8` | Full name |
| `team`, `opponent`, `is_home` | — | Team context |
| `position_group` | `pl.Utf8` | `"F"` (Forward: C/L/R) or `"D"` (Defenseman) |
| `toi_total_min`, `toi_ev_min`, `toi_pp_min`, `toi_pk_min` | `pl.Float64` | Time on ice by strength state (in decimal minutes) |
| `goals`, `assists`, `points`, `sog` | `pl.Int16` | Official counting stats (Stage 2 targets) |
| `ixg_all`, `ixg_5v5`, `icf_all` | `pl.Float64` | Individual Expected Goals ($ixG$) and shot attempts |
| `on_ice_xgf_5v5`, `on_ice_xga_5v5` | `pl.Float64` | On-ice 5v5 $xGF$ and $xGA$ while skater is on the ice |
| `game_score` | `pl.Float64` | Dom Luszczyszyn's single-game composite player impact metric |

### 2.4 `goalie_game_logs.parquet` (1 row per goalie appearance)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `game_id`, `game_date`, `start_time_utc`, `season` | — | Temporal & game keys |
| `player_id` | `pl.Int64` | NHL 7-digit goalie ID |
| `team`, `opponent`, `is_home` | — | Team context |
| `is_starter` | `pl.Boolean` | `True` if goalie started the game |
| `toi_min` | `pl.Float64` | Minutes played |
| `shots_against`, `saves`, `goals_against` | `pl.Int16` | Official boxscore goalie counts (Stage 2 targets) |
| `xga_all` | `pl.Float64` | Expected Goals Against faced based on shot quality |
| `gsax` | `pl.Float64` | Goals Saved Above Expected: $\text{GSAx} = \text{xGA}_{\text{all}} - \text{GA}$ |

---

## 3. Strict Anti-Leakage & Temporal Firewall Contract

Every feature transformation in `core/temporal.py` and `sports/<sport>/features/` MUST obey three sport-agnostic invariants designed to hold across **NHL**, **MLB** (including same-day doubleheaders), and **PGA** (multi-round tournaments):

### Invariant 1: Strict Left-Closed Shift (`shift(1)`) & Generalized Exposure Rates
For any entity $e \in \{\text{team}, \text{player}, \text{golfer}\}$ ordered by `(start_time_utc, event_seq, event_id)`, let $x_{e, t}$ be a post-event metric observed in appearance $t$. Any pre-event rolling statistic $R_{e, t}^{(W)}$ for window $W$ MUST be computed on the shifted series $x_{e, t-1}$:

$$\tilde{x}_{e, t} = \text{shift}(x_{e, t}, 1) \quad \text{over partition } e$$

In Polars, `core/temporal.py` provides both simple rolling/EWMA helpers and a **sport-agnostic exposure-normalized rate helper** (`pregame_exposure_rate`) that works for NHL per-60 TOI rates, MLB per-PA / per-9-IP rates, and PGA per-round Strokes Gained rates:

```python
def pregame_rolling_mean(col_name: str, window: int, group_by: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Float64)
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over(group_by)
    )

def pregame_exposure_rate(
    numerator_col: str,
    exposure_col: str,
    window: int,
    group_by: str,
    scale_factor: float = 1.0,  # 60.0 for NHL TOI min, 9.0 for MLB IP, 1.0 for MLB PA / PGA rounds
) -> pl.Expr:
    shifted_num = pl.col(numerator_col).cast(pl.Float64).shift(1).rolling_sum(window_size=window, min_samples=1).over(group_by)
    shifted_exp = pl.col(exposure_col).cast(pl.Float64).shift(1).rolling_sum(window_size=window, min_samples=1).over(group_by)
    return scale_factor * (shifted_num / pl.when(shifted_exp > 0).then(shifted_exp).otherwise(None))
```

### Invariant 2: Sub-Day Event Ordering (Doubleheader & Multi-Round Safety)
The core engine **never assumes an entity has at most one event per calendar date**:
* In **Baseball (MLB)**, teams frequently play same-day doubleheaders (`event_seq = 1` and `event_seq = 2` on the same `game_date`).
* In **Golf (PGA)**, golfers play 4 rounds across 4 days (and weather delays can push completion of two rounds onto the same calendar day).
* Therefore, all partitions are sorted strictly by `(start_time_utc, event_seq, event_id)`, and league/field baselines are computed strictly from events where `completed_time_utc < current_start_time_utc` (or `event_date < current_event_date` when intraday timestamps are unavailable).

### Invariant 3: Early-Season Cold-Start Bayesian Shrinkage
In games 1–10 of a new NHL season, rolling averages have high variance. For any rate metric $r_{e, t}$ observed over $n$ games in the current season, the pipeline blends the current-season rolling average $\bar{r}_{e, n}^{\text{curr}}$ with a regressed prior $\mu_{e}^{\text{prior}}$ (computed as $0.65 \cdot \bar{r}_{e}^{\text{prev\_season}} + 0.35 \cdot \mu_{\text{league}}$) using a pseudo-count shrinkage constant $k$ (default $k = 8$ games for teams, $k = 12$ games for skaters, $k = 15$ starts for goalies):

$$\hat{r}_{e, t}^{\text{shrunk}} = \frac{n \cdot \bar{r}_{e, n}^{\text{curr}} + k \cdot \mu_{e}^{\text{prior}}}{n + k}$$

When $n = 0$ (Opening Night or rookie debut), the feature cleanly evaluates to $\mu_{e}^{\text{prior}}$ with zero `NaN`s.

---

## 4. Stage 1 Feature Specification (Mandatory Calibrated Win Probability)

For each game $g$ between Home Team $H$ and Away Team $A$, the Stage 1 feature matrix (`data/gold/hockey/stage1_game_features.parquet`) computes features for $H$, $A$, and their **matchup differential** ($\Delta = \text{Home} - \text{Away}$):

### 4.1 Dynamic Elo Rating Features (`core/ratings.py`)
* **Pre-game Elo**: $E_H, E_A$, and $\Delta E = E_H + \text{HFA} - E_A$ (where base rating $= 1500$, $\text{HFA} \approx 24\text{--}35$ Elo points, $K = 8.0$).
* **Elo Win Probability**:
  $$P_{\text{Elo}}(\text{Home Win}) = \frac{1}{1 + 10^{-\Delta E / 400}}$$
* **Post-game Update Rule** (applied strictly *after* pre-game features are recorded):
  $$E_H^{\text{post}} = E_H + K \cdot M_{\text{MOV}} \cdot \left(S_H - P_{\text{Elo}}(\text{Home Win})\right)$$
  * Overtime/Shootout dampening: If the game ends in `"SO"`, $S_H = 0.60$ for the winner and $0.40$ for the loser (since shootouts are near-coin-flips).
  * Margin-of-victory multiplier: $M_{\text{MOV}} = \ln(|\text{GoalDiff}| + 1) \cdot \frac{2.2}{0.001 \cdot |\Delta E| + 2.2}$.
  * Off-season regression: Between seasons, $E_{\text{new\_season}} = 0.72 \cdot E_{\text{end\_prev}} + 0.28 \cdot 1500$ (configurable via `offseason_retention`).

### 4.2 Schedule, Rest & Travel Features (`core/schedule.py`)
* `rest_days_home`, `rest_days_away` (capped at 7 days), and `rest_advantage = rest_days_home - rest_days_away`.
* `is_b2b_home`, `is_b2b_away`: Binary flags (`rest_days == 1`), plus interaction `is_b2b_with_travel`.
* `games_last_4d_home`, `games_last_6d_home` (detects 3-in-4 and 4-in-6 fatigue spots).
* `travel_km_last_3d`, `tz_shift_hours`: Geodesic distance and timezone delta from previous game's venue.

### 4.3 Team 5v5 Process & Shot Quality Features (`sports/hockey/features/team.py`)
Computed over rolling windows $W \in \{5, 10, 25\}$ games plus EWMA ($\alpha = 0.12$), with Bayesian prior shrinkage:
* **5v5 Expected Goals Share**:
  $$\text{xGF\%}_{5v5} = \frac{\sum \text{xGF}_{5v5}}{\sum \text{xGF}_{5v5} + \sum \text{xGA}_{5v5}}$$
* **5v5 Pace & Suppression Rates (per 60 minutes of 5v5 TOI)**:
  $$\text{xGF/60}_{5v5} = 60 \cdot \frac{\sum \text{xGF}_{5v5}}{\sum \text{TOI}_{5v5}}, \quad \text{xGA/60}_{5v5} = 60 \cdot \frac{\sum \text{xGA}_{5v5}}{\sum \text{TOI}_{5v5}}$$
* **5v5 Shot-Attempt Territorial Shares**: Corsi Share ($\text{CF\%}_{5v5}$), Fenwick Share ($\text{FF\%}_{5v5}$), and High-Danger $xG$ Share ($\text{HD-xGF\%}_{5v5}$).
* **PDO (Luck / Regression Indicator)**:
  $$\text{PDO}_{5v5} = \left(\frac{\text{GF}_{5v5}}{\text{SOG}_{5v5}}\right) + \left(1 - \frac{\text{GA}_{5v5}}{\text{SA}_{5v5}}\right)$$
  Values significantly above $1.000$ indicate unsustainable shooting/save luck likely to regress; coupled with $\text{GF} - \text{xGF}$ ("Finishing Above Expected").

### 4.4 Special Teams Matchup Interaction Features
* Rolling Power Play efficiency ($\text{PP-xGF/60}$, $\text{PP\%}$) and Penalty Kill efficiency ($\text{PK-xGA/60}$, $\text{PK\%}$).
* **Special Teams Net Edge**:
  $$\text{ST\_Edge}_{H \text{ vs } A} = (\text{PP-xGF/60}_H - \text{PK-xGA/60}_A) \cdot \text{DrawRate}_H - (\text{PP-xGF/60}_A - \text{PK-xGA/60}_H) \cdot \text{DrawRate}_A$$

### 4.5 Starting Goalie Features (`sports/hockey/features/goalie.py`)
* Resolves confirmed/projected starter for game $g$ (historical: `is_starter == True` from pre-game lineup; live inference: confirmed starter or team's primary non-B2B starter heuristic).
* Rolling 10-start and 25-start **Shrunk GSAx per 60** ($\text{GSAx/60}$):
  $$\text{GSAx/60} = 60 \cdot \frac{\sum (\text{xGA} - \text{GA})}{\sum \text{TOI}_{\text{min}}}$$
* Starter fatigue: `goalie_starts_last_7d` and `goalie_days_rest`.

### 4.6 Active Roster Player Aggregation (`sports/hockey/features/roster.py`)
Bridges individual player availability (accounting for injuries, trades, and call-ups) into Stage 1 game predictions:
* For all active skaters $i \in \text{Roster}_{T, g}$ on team $T$ in game $g$:
  * Compute each skater's rolling shrunk pre-game $\widehat{\text{TOI}}_i$, individual offense rate $\widehat{\text{ixG/60}}_i$, on-ice impact $\widehat{\text{OnIce-xGD/60}}_i$, and $\widehat{\text{GameScore/60}}_i$.
  * Normalize projected TOI weights across Forwards ($\sum_{i \in F} w_i = 1$) and Defensemen ($\sum_{j \in D} w_j = 1$).
  * Compute **Lineup-Weighted Forward Strength**, **Lineup-Weighted Defense Strength**, and **Missing Regular TOI** (sum of rolling TOI of recent regulars absent from tonight's active roster).

---

## 5. Stage 2 Specification (Optional Player Matchup Stat Projections)

When Stage 2 is invoked (`--include-players` / `include_players=True`), the pipeline constructs a player-matchup feature table (`data/gold/hockey/stage2_player_features.parquet`) with 1 row per active player in the requested game.

### 5.1 Stage 2 Skater Input Features (`sports/hockey/features/player.py`)
1. **Individual Rolling Usage (5, 10, 25 games + EWMA, shrunk to position prior)**:
   * `rolling_toi_ev_min`, `rolling_toi_pp_min`, `rolling_shifts_per_game`
   * `team_injury_vacated_toi_f_or_d`: Vacated TOI at the skater's position group (`F` or `D`) from missing regulars, capturing usage bumps when top-line teammates are out.
2. **Individual Rolling Per-60 Production & Shot Quality**:
   * `rolling_sog_per_60`, `rolling_icf_per_60`, `rolling_ixg_per_60`
   * `rolling_goals_per_60`, `rolling_assists_per_60`, `rolling_points_per_60`
   * `career_shooting_pct_shrunk`: Career shooting percentage regressed toward position average ($G / \text{SOG}$) to separate true shooting talent from short-term streaks.
3. **Opponent & Goalie Matchup Context**:
   * Opponent rolling `xga_per_60_5v5`, `sa_per_60_5v5`, `pk_xga_per_60`, and `penalties_taken_per_60`.
   * Opponent starting goalie rolling `gsax_per_60_shrunk`.
4. **Stage 1 Macro Game Conditioning Signals**:
   * `stage1_team_win_prob`: Calibrated win probability for the player's team from Stage 1 (captures empty-net, score-effects, and trailing/leading usage scripts).
   * `stage1_expected_team_goals`: Stage 1 expected team goals for the player's team in this specific matchup.
   * `is_home`, `team_rest_days`, `is_b2b`.

### 5.2 Stage 2 Targets & Output Schema
For each active skater $i$ on both teams, Stage 2 outputs:
* `proj_toi_min`: Expected total Time on Ice (minutes).
* `proj_sog`: Expected Shots on Goal ($\hat{\mu}_{\text{SOG}}$).
* `proj_goals`: Expected Goals ($\hat{\mu}_{G}$).
* `proj_assists`: Expected Assists ($\hat{\mu}_{A}$).
* `proj_points`: Expected Points ($\hat{\mu}_{\text{PTS}} = \hat{\mu}_{G} + \hat{\mu}_{A}$).
* `prob_1plus_goal`: $1 - \exp(-\hat{\mu}_{G})$.
* `prob_1plus_point`: $1 - \exp(-\hat{\mu}_{\text{PTS}})$.

For each starting goalie $g$ on both teams, Stage 2 outputs:
* `proj_shots_against`: Expected Shots Against ($\hat{\mu}_{\text{SA}}$), anchored to opponent skater $\sum \hat{\mu}_{\text{SOG}}$.
* `proj_goals_against`: Expected Goals Against ($\hat{\mu}_{\text{GA}}$).
* `proj_saves`: $\hat{\mu}_{\text{SA}} - \hat{\mu}_{\text{GA}}$.
* `proj_save_pct`: $\hat{\mu}_{\text{SV}} / \max(\hat{\mu}_{\text{SA}}, 1)$.

### 5.3 Cross-Stage Coherence Reconciliation
After raw skater predictions $\hat{\mu}_{G, i}^{\text{raw}}$ are generated for team $T$, Stage 2 applies a damped reconciliation factor $\gamma \in [0, 1]$ (default $\gamma = 0.5$) toward Stage 1's expected regulation/OT team goals $\hat{\lambda}_T$:

$$\hat{\mu}_{G, i}^{\text{reconciled}} = \hat{\mu}_{G, i}^{\text{raw}} \cdot \left( \frac{\hat{\lambda}_T}{\sum_{j \in \text{Roster}_T} \hat{\mu}_{G, j}^{\text{raw}}} \right)^\gamma$$

This preserves relative player rankings while ensuring player projections scale realistically in high-scoring vs. low-scoring matchups.
