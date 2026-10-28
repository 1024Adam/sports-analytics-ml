# EPL Soccer Data Contracts & Feature Engineering Specification

> **Related Documents**: [System Architecture Blueprint](./architecture.md) | [Hockey Data & Feature Specification](./data-and-features.md) | [Implementation Roadmap](./roadmap.md)

## 1. Hybrid EPL Data Source Strategy

The soccer plugin (`sports/soccer/`) combines three complementary, publicly accessible data sources so the pipeline captures **official Premier League schedules/rosters/availability**, **research-grade expected goals ($xG, npxG, xA, PPDA$)**, and **historical match boxscore statistics**:

### Source A: Official Fantasy Premier League (FPL) / Premier League API (`https://fantasy.premierleague.com/api/`)
* **Role**: Canonical source of truth for the active 20 EPL clubs, fixture schedules, matchweek (`event`) assignments, UTC kickoff timestamps, squad registries, positional groups (`GKP`, `DEF`, `MID`, `FWD`), official injury/suspension flags (`status`, `chance_of_playing_next_round`), set-piece/penalty taker order, and official match-by-match player appearance stats.
* **Key Endpoints**:
  * `GET /api/bootstrap-static/`: 20 EPL teams, active squad elements (`id`, `web_name`, `first_name`, `second_name`, `team`, `element_type`), set-piece order (`penalties_order`, `corners_and_indirect_freekicks_order`, `direct_freekicks_order`), and availability metadata (`status` $\in \{\text{a}, \text{d}, \text{i}, \text{s}, \text{u}\}$, `chance_of_playing_next_round` $\in \{0, 25, 50, 75, 100\}$).
  * `GET /api/fixtures/`: All 380 fixtures in the season (`id`, `event`, `kickoff_time`, `team_h`, `team_a`, `team_h_score`, `team_a_score`, `finished`).
  * `GET /api/element-summary/{player_id}/` & `GET /api/event/{gw}/live/`: Player match logs (`minutes`, `starts`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `saves`, `penalties_saved`, `yellow_cards`, `red_cards`, `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`).

### Source B: Understat EPL Match & Player Datasets (`https://understat.com/league/EPL/`)
* **Role**: High-signal advanced shot-quality and tactical metrics per match and player across historical and active seasons, including non-penalty Expected Goals ($npxGF, npxGA$), total Expected Goals ($xGF, xGA$), Expected Assists ($xA$), $xGChain$, $xGBuildup$, Passes Allowed Per Defensive Action ($PPDA$), opponent $PPDA$, and deep completions (passes completed within 20 yards of goal excluding crosses).
* **Graceful Fallback**: If Understat lags immediately after final whistle, the reconciler falls back to FPL / Football-Data match and player stats and sets `has_understat_xg = False` without blocking daily predictions.

### Source C: Football-Data.co.uk Historical & Season CSVs (`https://www.football-data.co.uk/mmz4281/{SEASON}/E0.csv`)
* **Role**: Multi-season historical fixture results and team boxscore statistics (`FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, `HF`, `AF`, `HY`, `AY`, `HR`, `AR`) enabling fast, deterministic multi-season backfills.

---

## 2. Canonical Polars Parquet Schemas (Silver Layer)

All Silver tables are stored under `data/silver/soccer/` and strictly sorted by `(match_date, start_time_utc, event_seq, match_id)`.

### 2.1 `matches.parquet` (1 row per EPL match)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `sport` | `pl.Utf8` | Always `"soccer"` for the EPL plugin |
| `season` | `pl.Int32` | e.g., `20242025` (`2024` start year, `2025` end year) |
| `matchweek` | `pl.Int16` | Gameweek number (`1..38`) |
| `match_id` | `pl.Int64` | Canonical unique match ID (`event_id` alias) |
| `event_seq` | `pl.Int32` | Sub-day sequence (`1` default; supports double-gameweek ordering) |
| `match_date` | `pl.Date` | Calendar date of kickoff (UTC/UK) |
| `start_time_utc` | `pl.Datetime("us", "UTC")` | Exact kickoff UTC timestamp (canonical temporal sort key) |
| `home_team` | `pl.Utf8` | Canonical 3-letter club code (e.g., `"ARS"`, `"MCI"`, `"LIV"`) |
| `away_team` | `pl.Utf8` | Canonical 3-letter club code |
| `venue_timezone` | `pl.Utf8` | IANA timezone of stadium (`"Europe/London"`) |
| `game_state` | `pl.Utf8` | `"FINAL"` for completed matches; `"FUT"` for upcoming fixtures |
| `home_goals` | `pl.Int16` | Full-time goals scored by home club (`null` if `"FUT"`) |
| `away_goals` | `pl.Int16` | Full-time goals scored by away club (`null` if `"FUT"`) |
| `outcome_1x2` | `pl.Int8` | Multiclass target $y \in \{0, 1, 2\}$ (`0 = Home Win`, `1 = Draw`, `2 = Away Win`; `null` if `"FUT"`) |
| `home_win` | `pl.Int8` | Binary indicator (`1` if `home_goals > away_goals`, else `0`; `null` if `"FUT"`) |
| `is_draw` | `pl.Int8` | Binary indicator (`1` if `home_goals == away_goals`, else `0`; `null` if `"FUT"`) |
| `away_win` | `pl.Int8` | Binary indicator (`1` if `home_goals < away_goals`, else `0`; `null` if `"FUT"`) |

### 2.2 `team_match_logs.parquet` (2 rows per match — 1 Home, 1 Away)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `match_id` | `pl.Int64` | Joins to `matches.parquet` |
| `match_date` | `pl.Date` | Match date |
| `start_time_utc` | `pl.Datetime("us", "UTC")` | Temporal sort key |
| `season` | `pl.Int32` | Season identifier |
| `team` | `pl.Utf8` | Canonical 3-letter club code |
| `opponent` | `pl.Utf8` | Opponent 3-letter club code |
| `is_home` | `pl.Boolean` | `True` for home club, `False` for away |
| `goals_for`, `goals_against` | `pl.Int16` | Full-time goals scored and conceded |
| `shots_for`, `shots_against` | `pl.Int16` | Total shots attempted and faced |
| `sot_for`, `sot_against` | `pl.Int16` | Shots on target for and against |
| `xgf`, `xga` | `pl.Float64` | Total Expected Goals for and against |
| `npxgf`, `npxga` | `pl.Float64` | Non-Penalty Expected Goals for and against |
| `ppda`, `opp_ppda` | `pl.Float64` | Passes Allowed Per Defensive Action (lower = higher pressing intensity) |
| `deep_completions`, `opp_deep_completions` | `pl.Int16` | Completed passes within 20m of goal (excluding crosses) |
| `corners_for`, `corners_against` | `pl.Int16` | Corners won and conceded |
| `fouls_for`, `fouls_against` | `pl.Int16` | Fouls committed and drawn |
| `yellow_cards`, `red_cards` | `pl.Int16` | Disciplinary cards received |
| `has_understat_xg` | `pl.Boolean` | `True` when Understat $npxG / PPDA$ was joined; `False` on fallback |

### 2.3 `outfield_match_logs.parquet` (1 row per outfield player appearance)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `match_id`, `match_date`, `start_time_utc`, `season` | — | Temporal & match keys |
| `player_id` | `pl.Int64` | Canonical player identifier |
| `player_name` | `pl.Utf8` | Full normalized player name |
| `team`, `opponent`, `is_home` | — | Club context |
| `position_group` | `pl.Utf8` | `"FWD"`, `"MID"`, or `"DEF"` |
| `is_starter` | `pl.Boolean` | `True` if in starting XI |
| `minutes` | `pl.Float64` | Minutes played ($\in [0, 90+]$) |
| `goals`, `assists`, `shots`, `sot` | `pl.Int16` | Official counting stats (Stage 2 targets) |
| `key_passes` | `pl.Int16` | Shot-creating passes |
| `xg`, `npxg`, `xa` | `pl.Float64` | Individual $xG$, non-penalty $xG$, and Expected Assists ($xA$) |
| `xg_chain`, `xg_buildup` | `pl.Float64` | Possession-chain $xG$ involvement metrics |
| `yellow_cards`, `red_cards` | `pl.Int16` | Cards received |

### 2.4 `gk_match_logs.parquet` (1 row per goalkeeper appearance)
| Column | Polars Type | Description |
| :--- | :--- | :--- |
| `match_id`, `match_date`, `start_time_utc`, `season` | — | Temporal & match keys |
| `player_id` | `pl.Int64` | Canonical goalkeeper ID |
| `player_name` | `pl.Utf8` | Goalkeeper name |
| `team`, `opponent`, `is_home` | — | Club context |
| `is_starter` | `pl.Boolean` | `True` if starting goalkeeper |
| `minutes` | `pl.Float64` | Minutes played |
| `shots_on_target_against`, `saves`, `goals_against` | `pl.Int16` | Official GK boxscore counts |
| `clean_sheet` | `pl.Int8` | `1` if `minutes >= 60` and `goals_against == 0`, else `0` |
| `xga_faced` | `pl.Float64` | Expected Goals Against faced |
| `gsaa` | `pl.Float64` | Goals Saved Above Average/Expected: $\text{GSAA} = \text{xGA}_{\text{faced}} - \text{GA}$ |

---

## 3. Temporal Firewall, Exposure Basis & Shrinkage Priors

### 3.1 Per-90 Exposure Rates (`scale_factor = 90.0`)
All rate features for EPL teams and players are normalized per 90 minutes using `pregame_exposure_rate(..., scale_factor=90.0)` from `src/sports_ml/core/temporal.py` with a strict `.shift(1)` over `(start_time_utc, event_seq, match_id)`:

$$\text{Rate}_{90}(x) = 90.0 \cdot \frac{\sum_{t-W}^{t-1} x}{\sum_{t-W}^{t-1} \text{minutes}}$$

### 3.2 Early-Season & Low-Sample Bayesian Shrinkage Constants ($k$)
To eliminate early-season variance (Matchweeks 1–8) and handle promoted clubs or low-minute squad players, rolling metrics are blended with prior baselines via `bayesian_shrinkage(curr, n, prior, k)`:
* **Team Rolling Efficiency ($npxGF/90, npxGA/90, PPDA$)**: $k_{\text{team}} = 6.0$ matches (prior $= 0.65 \cdot \mu_{\text{prev\_season}} + 0.35 \cdot \mu_{\text{league}}$; promoted clubs use a promoted-club prior $\mu_{\text{promoted}}$).
* **Outfield Player Per-90 Rates ($npxG/90, xA/90, \text{Shots}/90$)**: $k_{\text{outfield}} = 9.0$ ninety-minute equivalents ($810$ minutes), regressed toward positional priors (`FWD`, `MID`, `DEF`).
* **Goalkeeper $\text{GSAA}/90$ & $\text{SV\%}$**: $k_{\text{gk}} = 12.0$ starts, regressed toward league-average goalkeeper performance ($\text{GSAA}/90 = 0.0$, $\text{SV\%} = 0.710$).

### 3.3 Dynamic 3-Way (`1X2`) Elo & Promotion/Relegation Offseason Handling
* **Base Rating & HFA**: `base_rating = 1500.0`, `home_field_advantage = 48.0` Elo points, `k_factor = 18.0`, `offseason_retention = 0.75`.
* **Promotion / Relegation Handler**: At each season boundary, returning top-flight clubs regress toward `1500.0` ($0.75 \cdot R_{\text{end}} + 0.25 \cdot 1500$), whereas newly promoted EFL Championship clubs initialize at `promoted_team_rating = 1390.0`.
* **Analytical 3-Way (`1X2`) Elo Probabilities**:
  Given expected score $E_H = \frac{1}{1 + 10^{-\Delta E / 400}}$ and baseline draw parameter $\nu \in [0, 0.5)$ (default $\nu \approx 0.26$ for EPL):
  $$P_{\text{Elo}}(\text{Draw}) = \nu \cdot 4 \cdot E_H \cdot (1 - E_H)$$
  $$P_{\text{Elo}}(\text{Home Win}) = E_H - \frac{1}{2} P_{\text{Elo}}(\text{Draw}), \quad P_{\text{Elo}}(\text{Away Win}) = (1 - E_H) - \frac{1}{2} P_{\text{Elo}}(\text{Draw})$$
  By construction, $P_{\text{Elo}}(\text{Home Win}) + P_{\text{Elo}}(\text{Draw}) + P_{\text{Elo}}(\text{Away Win}) = 1.0$, and $P_{\text{Elo}}(\text{Draw})$ peaks at $\nu$ when $\Delta E = 0$ and decays smoothly as $|\Delta E|$ grows.

---

## 4. Stage 1 & Stage 2 Target Summary

### 4.1 Stage 1 Targets & Evaluation Metrics
* **Primary Target**: `outcome_1x2` $\in \{0 (\text{Home Win}), 1 (\text{Draw}), 2 (\text{Away Win})\}$ with calibrated probabilities $(P_H, P_D, P_A)$ summing strictly to $1.0$.
* **Auxiliary Targets**: Full-time team goals $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ with Dixon-Coles low-score correlation parameter $\rho$.
* **Evaluation Metrics**:
  * **Ranked Probability Score (RPS)**: Ordinal multi-outcome error across cumulative distributions $F_1 = P_H, F_2 = P_H + P_D$:
    $$\text{RPS} = \frac{1}{2} \sum_{k=1}^{2} \left( F_k - O_k \right)^2$$
  * **3-Class Log Loss**, **Multi-Class Brier Score**, **Multiclass ECE**, and **Accuracy**, benchmarked against uniform (`1/3`), empirical EPL prior (`0.45 H / 0.24 D / 0.31 A`), and Elo-1X2 baselines.

### 4.2 Stage 2 Targets & Cross-Stage Coherence Reconciliation
* **Outfield Projections**: `proj_minutes` ($\in [0, 90]$), `proj_goals`, `proj_assists`, `proj_goal_involvements` ($G+A$), `proj_shots`, `proj_sot`, `prob_1plus_goal`, `prob_1plus_assist`, and `prob_1plus_gi`.
* **Goalkeeper Projections**: `proj_saves`, `proj_goals_against`, `proj_save_pct`, and `prob_clean_sheet`.
* **Cross-Stage Coherence ($\gamma = 0.50$)**: Outfield player `proj_goals` and `proj_assists` are soft-scaled toward Stage 1's $\hat{\lambda}_{\text{team}}$ (with EPL assist-to-goal ratio $\approx 0.72$).

---

## 5. Phase 4 Extensions: Market Benchmarking, Cross-Competition Fatigue & Shot Micro-Features

### 5.1 Evaluation-Only 1X2 Closing Odds (`market_*` Firewall, `Milestone 13`)
* Extracted directly from the cached `Football-Data.co.uk` `E0_{code}.csv.gz` files (`PSCH`/`PSCD`/`PSCA` Pinnacle closing odds, with fallback to `B365CH`/`B365CD`/`B365CA` and opening lines) at zero additional network cost.
* Stored in `matches.parquet` as `market_home_decimal_odds`, `market_draw_decimal_odds`, `market_away_decimal_odds`, and power-de-vigged `market_prob_home`, `market_prob_draw`, `market_prob_away`.
* **Hard Feature Firewall**: Never included in `STAGE1_FEATURE_COLUMNS` or `STAGE2_FEATURE_COLUMNS`. Used strictly in `sports-ml backtest` and `sports-ml score-predictions` to compare model **RPS / Log Loss / Brier** against the closing market and evaluate simulated ROI when $|P_{\text{model}} - P_{\text{market}}| \ge 5\%$.

### 5.2 Cross-Competition Schedule & Squad Fatigue (`Milestone 14`)
* Non-league fixtures (**UEFA Champions League**, **Europa League**, **Conference League**, **FA Cup**, **EFL Cup**) for the 20 EPL clubs are stored in `data/silver/soccer/club_all_competitions_schedule.parquet`.
* **Fatigue-Only Usage**: Used strictly to compute `true_all_comp_rest_days`, `had_midweek_european_match`, `european_travel_km_last_7d`, `all_comp_matches_last_14d`, `cup_extra_time_last_7d`, and `active_xi_14d_all_comp_minutes`.
* **Tactical Isolation**: Rolling EPL $npxGF/90$, $npxGA/90$, $PPDA$, and club Elo ratings filter exclusively on `competition_type == "EPL"` so cup ties against lower-division or foreign opponents never distort domestic tactical ratings.

### 5.3 Shot-Coordinate & Situation Micro-Features (`Milestone 15`)
* Extracted from Understat match shot arrays (`situation`, $(X, Y)$ coordinates, `lastAction`, minute, and live score-state) into `data/silver/soccer/team_shot_micro_logs.parquet`:
  * **Open-Play vs. Set-Piece $xG$ Split (`open_play_xgf`, `set_piece_xgf`, `open_play_xga`, `set_piece_xga`)**: Separates sustainable open-play chance creation from corner/free-kick efficiency and set-piece defensive vulnerability.
  * **Game-State Neutral $npxG$ (`tied_state_npxgd_per_90`)**: Strips out late-game score-effect distortion when a club trails by $2+$ goals.
  * **Spatial Box & Throughball Danger (`box_shot_share`, `throughball_xgf`)**: Captures central 18-yard box penetration vs. low-probability long shots.

