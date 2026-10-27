# Phased Implementation Roadmap

> **Related Documents**: [System Architecture Blueprint](./architecture.md) | [Data & Feature Specification](./data-and-features.md)

This roadmap breaks the `sports-analytics-ml` platform into sequential, testable milestones across two phases (**Phase 1: Core Engine & NHL Hockey**, Milestones 1–6; **Phase 2: English Premier League (EPL) Soccer**, Milestones 7–11) designed for execution by the `sports-ml-developer` agent. Each milestone has explicit file deliverables, acceptance criteria, and mandatory `pytest` anti-leakage gates.

---

## Milestone 1: Project Foundation & Sport-Agnostic Core Engine (`core/`)
**Status**: `[x] Completed`

### Goal
Establish the Python 3.11+ package structure (`pyproject.toml`, `src/sports_ml/`), canonical data contracts, and the sport-agnostic time-series and rating primitives in `src/sports_ml/core/`.

### Deliverables
- [x] `.gitignore`: Strictly excludes `data/`, `artifacts/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, and `.pytest_cache/`.
- [x] `pyproject.toml` and `uv.lock` configured with `polars`, `pyarrow`, `pydantic`, `httpx`, `tenacity`, `lightgbm`, `scikit-learn`, `scipy`, `typer`, `rich`, `pytest`, `pytest-cov`, `mypy`, and `ruff` (with rule sets `E`, `F`, `I`, `B`, `UP`, `SIM`, `NPY`), plus `src/sports_ml/config.py` defining deterministic `RANDOM_SEED = 42`.
- [x] `.github/workflows/ci.yml`: Fast, offline GitHub Actions CI workflow triggered on `push` and `pull_request` to `main` (with `permissions: contents: read`) running a tracked-binary/data guard check, `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml`.
- [x] `src/sports_ml/core/contracts.py`: `SportAdapter` protocol supporting both `head_to_head` (NHL, MLB, PGA H2H) and `multi_entrant_field` (PGA tournament field) competition formats, canonical event/participant schema validators, and starter/venue modifier hooks.
- [x] `src/sports_ml/core/temporal.py`: Leak-free Polars window helpers (`pregame_rolling_mean`, `pregame_rolling_sum`, `pregame_ewma`, and generalized `pregame_exposure_rate` supporting per-60 TOI, per-PA/per-9 IP, and per-round Strokes Gained) enforcing `.shift(1)` ordered by `(start_time_utc, event_seq, event_id)`.
- [x] `src/sports_ml/core/shrinkage.py`: Early-season / low-sample Bayesian prior blending (`(n * curr + k * prior) / (n + k)`).
- [x] `src/sports_ml/core/ratings.py`: Dynamic rating engine supporting pairwise Elo/Glicko updates (with Home-Field Advantage, Margin-of-Victory multiplier, OT/Shootout dampening, starter adjustment hooks, and between-season regression) and extensible multi-entrant field skill ratings.
- [x] `src/sports_ml/core/schedule.py`: Rest days, back-to-back (`is_b2b`), same-day doubleheader turnaround (`event_seq`), `k`-in-`m` nights density, geodesic venue haversine distance, and timezone shift calculator.
- [x] `src/sports_ml/core/validation.py`: Walk-forward expanding and sliding window time-series cross-validator grouped by chronological event timestamp.

### Verification & Acceptance Criteria
- [x] `tests/core/test_temporal_leakage.py`: **Future-Mutation Invariance Test** — mutating all target and box-score columns for events at index $\ge t$ (including same-day doubleheader Game 2 or multi-round tournament Round $r+1$) produces **bit-for-bit identical** features at index $< t$.
- [x] `tests/core/test_ratings.py`: Verify Elo zero-sum conservation, home-field shift, shootout dampening, and off-season mean regression.
- [x] `tests/core/test_schedule.py`: Verify B2B, same-day doubleheader, and 3-in-4 calculations across month/year boundaries and venue travel distances.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 2: Hybrid NHL Ingestion & Silver Lakehouse (`sports/hockey/ingestion/`)
**Status**: `[x] Completed`

### Goal
Build resilient, cached ingestion pipelines for the **Official NHL Web API** and **MoneyPuck CSV datasets**, normalizing and reconciling them into canonical Silver Parquet tables.

### Deliverables
- [x] `src/sports_ml/sports/hockey/constants.py`: NHL 32-team registry, historical tricode normalization (`ARI` $\rightarrow$ `UTA`, `S.J` $\rightarrow$ `SJS`, `L.A` $\rightarrow$ `LAK`, `N.J` $\rightarrow$ `NJD`, `T.B` $\rightarrow$ `TBL`), and arena latitude/longitude/timezone coordinates.
- [x] `src/sports_ml/sports/hockey/ingestion/nhl_api.py`: `httpx` client with deterministic disk caching (`data/raw/nhl_api/`) and `tenacity` retry/backoff for schedules, game boxscores, and team rosters. Supports permanent caching for `FINAL`/`OFF` games and incremental delta fetching (`--current-season` / `--start-date` / `--end-date`) for newly completed games and upcoming `FUT` schedules.
- [x] `src/sports_ml/sports/hockey/ingestion/moneypuck.py`: Downloader and Polars parser for MoneyPuck team, skater, and goalie game-by-game CSVs (`5on5`, `5on4`, `4on5`, `all`), with TTL/ETag refresh for current-season-to-date CSV snapshots.
- [x] `src/sports_ml/sports/hockey/ingestion/lineups.py`: **2-Tier Live Pre-Game Lineup, Starting Goalie & Injury Resolver**:
  - **Tier 1**: Official NHL API pre-game/warmup dressed lineup & starter (`/v1/gamecenter/{GAME_ID}/boxscore` + `/v1/roster/{TEAM}/current`).
  - **Tier 2**: Daytime morning-skate starting goalies (`Confirmed` / `Expected`), projected line combinations, and injury exclusions (e.g., DailyFaceoff feed reconciled to NHL 7-digit `player_id`s) when predicting hours before puck drop.
- [x] `src/sports_ml/sports/hockey/ingestion/reconciler.py`: Upserts and joins NHL API and MoneyPuck records into the 4 canonical Silver Parquet tables (`games.parquet`, `team_game_logs.parquet`, `skater_game_logs.parquet`, `goalie_game_logs.parquet`), with graceful fallback when MoneyPuck lags by a day.

### Verification & Acceptance Criteria
- [x] `tests/sports/hockey/test_ingestion.py`: Test JSON/CSV parsing, tricode normalization, schema adherence, incremental upsert (transitioning a game from `FUT` to `FINAL` as the current season progresses), 2-Tier pre-game lineup/goalie/injury resolution (Tier 1 warmup payload vs. Tier 2 daytime morning-skate payload), and fallback handling using deterministic local fixtures.
- [x] Verify Silver tables enforce uniqueness on `(game_id)` for games, `(game_id, team)` for team logs, and `(game_id, player_id)` for player logs.

---

## Milestone 3: Hockey Pre-Game Feature Engineering (`sports/hockey/features/`)
**Status**: `[x] Completed`

### Goal
Construct the complete pre-game feature suite for **Stage 1 (Game Outcome)** and **Stage 2 (Player Matchup Projections)** in `data/gold/hockey/`.

### Deliverables
- [x] `src/sports_ml/sports/hockey/features/team.py`: Rolling 5, 10, 25-game and EWMA features for 5v5 $xGF\%$, $xGF/60$, $xGA/60$, Corsi ($CF\%$), Fenwick ($FF\%$), High-Danger $xG\%$, PDO luck/regression signal, and Special Teams Net Edge (PP vs. PK).
- [x] `src/sports_ml/sports/hockey/features/goalie.py`: Starting goalie resolver and rolling shrunk $\text{GSAx/60}$, $\text{SV\%}$, and goalie workload/fatigue metrics.
- [x] `src/sports_ml/sports/hockey/features/roster.py`: Active roster aggregator weighting individual skater rolling $\text{ixG/60}$, $\text{OnIce-xGD/60}$, and $\text{GameScore/60}$ by projected TOI share for Forwards and Defensemen, plus `missing_regular_toi`.
- [x] `src/sports_ml/sports/hockey/features/player.py`: Stage 2 individual player-matchup feature builder (rolling EV/PP TOI, vacated injury TOI, per-60 rates, shrunk career shooting %, opponent defensive suppression, and opponent starting goalie GSAx).

### Verification & Acceptance Criteria
- [x] `tests/sports/hockey/test_hockey_features.py`:
  - **Strict Anti-Leakage Test**: Verify that changing tonight's game goals, $xG$, or player points does not alter any pre-game team, goalie, roster, or player feature for tonight's game.
  - **Cold-Start Test**: Verify Game 1 of a season and a debuting rookie/goalie produce finite, non-null shrunk features.
  - **Roster Weighting Test**: Verify TOI weights normalize cleanly and reflect lineup changes when a top-line player is removed from the active roster.

---

## Milestone 4: Stage 1 — Mandatory Calibrated Win Probability Pipeline
**Status**: `[x] Completed`

### Goal
Implement the mandatory Stage 1 game outcome model, probability calibration wrapper, auxiliary team-goals environment estimator, and walk-forward backtesting harness.

### Deliverables
- [x] `src/sports_ml/core/calibration.py`: Out-of-fold Platt (Sigmoid) and Isotonic probability calibrators.
- [x] `src/sports_ml/core/evaluation.py`: Evaluation suite computing **Log Loss**, **Brier Score**, Expected Calibration Error (ECE), accuracy, and benchmark comparisons against coin-flip, home-team baseline, and Elo-only baseline.
- [x] `src/sports_ml/sports/hockey/models/stage1_game.py`:
  - Trained Elo baseline + LightGBM classifier producing calibrated $P(\text{Home Win})$ and $P(\text{Away Win}) = 1 - P(\text{Home Win})$.
  - Auxiliary Poisson/Tweedie team-goals regressor estimating $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ to provide macro pace context for Stage 2.

### Verification & Acceptance Criteria
- [x] `tests/sports/hockey/test_stage1_model.py`:
  - Verify predicted probabilities lie strictly in $(0, 1)$ and $P(\text{Home Win}) + P(\text{Away Win}) = 1.0$.
  - Verify walk-forward training never fits the GBDT or calibrator on validation/test dates.
  - Verify calibrated model achieves lower Brier Score / Log Loss than a constant home-win-rate baseline on synthetic directional data.

---

## Milestone 5: Stage 2 — Optional Player Matchup Stat Projections
**Status**: `[x] Completed`

### Goal
Implement the optional Stage 2 player stat projection models for both teams' skaters and goalies, conditioned on Stage 1's game outcome and scoring environment.

### Deliverables
- [x] `src/sports_ml/sports/hockey/models/stage2_player.py`:
  - **Skater TOI Projector**: Estimates $\widehat{\text{TOI}}_{\text{ev}}$ and $\widehat{\text{TOI}}_{\text{pp}}$ for active roster skaters.
  - **Skater Count/Rate Models**: LightGBM Poisson/Tweedie regressors conditioned on player rolling rates, opponent defense/goalie features, and Stage 1 context ($P(\text{Win})$, $\hat{\lambda}_{\text{team}}$) predicting `proj_goals`, `proj_assists`, `proj_points`, `proj_sog`, `prob_1plus_goal`, and `prob_1plus_point`.
  - **Goalie Projection Model**: Predicts `proj_shots_against`, `proj_goals_against`, `proj_saves`, and `proj_save_pct`.
  - **Cross-Stage Coherence Reconciler**: Soft-scales team skater goal totals toward Stage 1's $\hat{\lambda}_{\text{team}}$.

### Verification & Acceptance Criteria
- [x] `tests/sports/hockey/test_stage2_player.py`:
  - Verify all predicted counts and TOI are non-negative ($\ge 0$) and milestone probabilities lie in $[0, 1]$.
  - Verify **Stage-Conditioning Monotonicity**: Increasing Stage 1's expected team goals $\hat{\lambda}_{\text{team}}$ or facing a weaker opponent goalie (lower GSAx) increases reconciled skater goal/point projections.
  - Verify Stage 2 can be toggled completely off (`include_players=False`) with zero overhead when only Stage 1 win probability is requested.

---

## Milestone 6: Unified Two-Stage Predictor Service & CLI (`predictor.py` & `cli.py`)
**Status**: `[x] Completed`

### Goal
Wire Stage 1 and Stage 2 into a cohesive inference API, artifact registry, and `sports-ml` CLI tool with Rich terminal output.

### Deliverables
- [x] `src/sports_ml/core/registry.py`: Model bundle save/load with `metadata.json` fingerprinting (ordered feature names, Polars dtypes, SHA-256 schema hash, training date range, git SHA, and validation metrics) that validates feature compatibility before inference.
- [x] `src/sports_ml/sports/hockey/models/predictor.py`: `HockeyMatchupPredictor` supporting `predict_matchup(..., include_players: bool = False)` and `predict_date(..., include_players: bool = False)`, surfacing pre-prediction confidence warnings (confirmed vs. projected starting goalie, data freshness, rookie prior flags), and automatically appending predictions to the live audit ledger (`data/predictions/hockey_predictions.parquet`).
- [x] `src/sports_ml/cli.py`: `typer` CLI implementing:
  - Streamlined auto-chaining: `train` automatically builds/updates Gold features if missing or stale; `predict` connects live by default (with `--offline` opt-out) to run incremental `--current-season` ingestion, resolve live 2-Tier pre-game lineups/goalies/injuries, and refresh features before predicting.
  - Granular developer & audit subcommands: `ingest`, `features build`, `backtest` (R&D evaluation), `train`, `predict` (with `--include-players`, `--matchup`, and `--offline` flags), and `score-predictions` (evaluates the live `data/predictions/` ledger against completed games).
- [x] `README.md`: End-to-end quickstart guide, CLI usage examples, and architecture overview.

### Verification & Acceptance Criteria
- [x] Verify `registry.py` raises a clear schema-mismatch error if a feature column is missing or reordered at inference time.
- [x] End-to-end CLI integration test running `ingest` $\rightarrow$ `features` $\rightarrow$ `train` $\rightarrow$ `backtest` $\rightarrow$ `predict` $\rightarrow$ `score-predictions` (both default Stage 1 only and Stage 1 + Stage 2 with `--include-players`) on fixture data.
- [x] Full `pytest --cov=sports_ml`, `ruff check .`, `ruff format --check .`, and `mypy src tests` pass cleanly.

---

# Phase 2: English Premier League (EPL) Soccer Plugin (`sports/soccer/`)

Phase 2 extends `sports-analytics-ml` to **English Premier League (EPL) Soccer (`--sport soccer`)** with a full two-stage prediction pipeline:
1. **Stage 1 — Mandatory Calibrated 3-Way (`1X2`) Match Outcome & Goals Environment Model**: Produces calibrated probabilities $P(\text{Home Win})$, $P(\text{Draw})$, and $P(\text{Away Win})$ summing strictly to $1.0$, alongside expected goals $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$.
2. **Stage 2 — Optional Outfield & Goalkeeper Matchup Stat Projections**: Conditioned on Stage 1's game script and scoring environment, projects outfield player minutes, goals, assists, goal involvements ($G+A$), shots, shots on target, milestone probabilities ($P(\text{G} \ge 1), P(\text{A} \ge 1), P(\text{G+A} \ge 1)$), and goalkeeper saves, goals against, save %, and clean-sheet probability $P(\text{CS})$.

---

## Milestone 7: Core Multi-Outcome (`1X2`) Extensions & Soccer Plugin Foundation (`core/` & `sports/soccer/`)
**Status**: `[x] Completed`

### Goal
Extend the sport-agnostic `src/sports_ml/core/` engine to support 3-way (`1X2`: Home Win, Draw, Away Win) competition outcomes, promotion/relegation Elo priors, Ranked Probability Score (RPS) evaluation, and per-90 exposure rates, while establishing the EPL data specification and `SoccerAdapter`.

### Deliverables
- [x] `docs/soccer-data-and-features.md`: Comprehensive EPL data & feature specification defining Silver Parquet schemas (`matches.parquet`, `team_match_logs.parquet`, `outfield_match_logs.parquet`, `gk_match_logs.parquet`), Bayesian shrinkage constants $k$, promoted-club Elo priors, and Stage 1 (`1X2` + $\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}}$) / Stage 2 targets.
- [x] `src/sports_ml/core/ratings.py`: Extend `EloConfig` and `PairwiseEloEngine` to support:
  - Analytical 3-way (`1X2`) Elo win/draw/loss probabilities (`elo_home_win_prob`, `elo_draw_prob`, `elo_away_win_prob` summing to `1.0`, parameterized by a baseline draw parameter $\nu$ where $P(\text{Draw})$ peaks at $\Delta \text{Elo} = 0$ and decays with rating disparity).
  - **Promotion / Relegation Offseason Handler**: Between seasons, returning clubs regress toward `base_rating` (`1500.0`), while newly promoted EFL Championship clubs initialize at a configurable `promoted_team_rating` (default `1390.0`).
- [x] `src/sports_ml/core/calibration.py`: Add `MulticlassProbabilityCalibrator` supporting out-of-fold 3-way (`1X2`) temperature scaling / Dirichlet-Platt / isotonic calibration with strict simplex normalization ($\sum_{c \in \{H, D, A\}} P_c = 1.0$, $P_c \in (\epsilon, 1-\epsilon)$).
- [x] `src/sports_ml/core/evaluation.py`: Add `MulticlassEvaluationMetrics`, `MulticlassBenchmarkReport`, `ranked_probability_score` (**RPS** — ordinal multi-outcome error metric over $(H, D, A)$), 3-class **Log Loss**, multi-class **Brier Score**, multiclass **ECE**, accuracy, and baseline comparisons (uniform `1/3`, empirical EPL prior `~45% H / 24% D / 31% A`, and Elo-1X2 baseline).
- [x] `src/sports_ml/core/temporal.py`: Document and verify `pregame_exposure_rate(..., scale_factor=90.0)` for per-90-minute soccer rate normalization.
- [x] `src/sports_ml/sports/soccer/constants.py`: Historical & current EPL club registry (all current and recent promoted/relegated clubs with canonical 3-letter codes `ARS`, `AVL`, `BOU`, `BRE`, `BHA`, `CHE`, `CRY`, `EVE`, `FUL`, `IPS`, `LEI`, `LIV`, `MCI`, `MUN`, `NEW`, `NFO`, `SOU`, `TOT`, `WHU`, `WOL`, `LUT`, `BUR`, `SHU`, `LEE`, `SUN`, etc.), cross-source team name alias normalizer (FPL $\leftrightarrow$ Understat $\leftrightarrow$ Football-Data.co.uk), and UK stadium latitude/longitude/timezone (`Europe/London`) coordinates.
- [x] `src/sports_ml/sports/soccer/adapter.py`: `SoccerAdapter` (`sport_slug = "soccer"`, `topology = "head_to_head"`) implementing `SportAdapter` protocol.

### Verification & Acceptance Criteria
- [x] `tests/core/test_multiclass_calibration_eval.py`: Verify 3-way `1X2` probabilities strictly sum to `1.0` in $(0, 1)$, verify Ranked Probability Score (RPS) penalizes distant outcome errors ($H$ vs. $A$) strictly more than adjacent errors ($H$ vs. $D$), and verify promoted teams receive `promoted_team_rating` upon season transition.
- [x] `tests/sports/soccer/test_adapter_constants.py`: Verify canonical club tricode normalization across FPL, Understat, and Football-Data naming variants, and verify `SoccerAdapter` event/participant schema validation.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 8: Hybrid EPL Ingestion & Silver Lakehouse (`sports/soccer/ingestion/`)
**Status**: `[x] Completed`

### Goal
Build resilient, cached ingestion pipelines for the **Official Fantasy Premier League (FPL) / Premier League API**, **Understat**, and **Football-Data.co.uk**, normalizing and reconciling them into canonical Silver Parquet tables under `data/silver/soccer/`.

### Deliverables
- [x] `src/sports_ml/sports/soccer/ingestion/fpl_api.py`: `httpx` client with deterministic disk caching (`data/raw/fpl_api/`) and `tenacity` retry/backoff for:
  - `/api/bootstrap-static/`: 20 active EPL clubs, player registry, positions (`GKP`, `DEF`, `MID`, `FWD`), set-piece/penalty taker order (`penalties_order`, `corners_and_indirect_freekicks_order`, `direct_freekicks_order`), and official injury/suspension flags (`status`, `chance_of_playing_next_round`).
  - `/api/fixtures/`: Full 380-match season schedule, matchweek (`event`) assignments, UTC kickoff timestamps, finished flags, and official scores.
  - `/api/element-summary/{player_id}/` & `/api/event/{gw}/live/`: Match-by-match player appearances (`minutes`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `saves`, `penalties_saved`, `yellow_cards`, `red_cards`, `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`).
- [x] `src/sports_ml/sports/soccer/ingestion/understat.py`: Client and Polars parser (`data/raw/understat/`) for historical & active-season EPL match and player shot-quality metrics: non-penalty expected goals ($npxG$, $npxGA$), total $xG$, expected assists ($xA$), $xGChain$, $xGBuildup$, shots, key passes, Passes Allowed Per Defensive Action ($PPDA$), and deep completions.
- [x] `src/sports_ml/sports/soccer/ingestion/football_data.py`: Historical season CSV downloader/parser (`data/raw/football_data/`, `E0.csv` per season) providing team shots (`HS`, `AS`), shots on target (`HST`, `AST`), corners (`HC`, `AC`), fouls (`HF`, `AF`), cards (`HY`, `AY`, `HR`, `AR`), and half-time/full-time results across multi-season backfills.
- [x] `src/sports_ml/sports/soccer/ingestion/lineups.py`: **2-Tier Live Pre-Match Lineup, Starting GK & Injury/Suspension Resolver**:
  - **Tier 1**: Official pre-match starting XI (`11` starters + `9` bench substitutes) posted ~60–75 minutes before kickoff.
  - **Tier 2**: Daytime FPL official injury/suspension status (`status` $\in \{\text{a}, \text{d}, \text{i}, \text{s}, \text{u}\}$, `chance_of_playing_next_round` $\in \{0, 25, 50, 75, 100\}$) + projected starting goalkeeper and outfield XI based on recent starts when predicting days/hours before kickoff.
- [x] `src/sports_ml/sports/soccer/ingestion/reconciler.py` & `health.py`: Cross-source entity reconciler joining FPL, Understat, and Football-Data records into the 4 canonical Silver Parquet tables (`matches.parquet`, `team_match_logs.parquet`, `outfield_match_logs.parquet`, `gk_match_logs.parquet`) with graceful fallback (`has_understat_xg = False`) when Understat lags post-match, plus `check_soccer_data_sources` for `sports-ml check-sources --sport soccer`.

### Verification & Acceptance Criteria
- [x] `tests/sports/soccer/test_ingestion.py`: Test FPL JSON, Understat payload, and Football-Data CSV parsing, club and player ID reconciliation, incremental matchweek upsert (`FUT` $\rightarrow$ `FINAL`), 2-Tier pre-match lineup/injury/suspension resolution, and graceful fallback when Understat lags, using 100% offline deterministic fixtures.
- [x] Verify Silver tables enforce uniqueness on `(match_id)` for matches, `(match_id, team)` for team logs, and `(match_id, player_id)` for player logs.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 9: EPL Pre-Match Feature Engineering (`sports/soccer/features/`)
**Status**: `[x] Completed`

### Goal
Construct the strictly leak-free pre-match feature suite for **Stage 1 (Match `1X2` Outcome & Goals Environment)** and **Stage 2 (Outfield & Goalkeeper Matchup Projections)** in `data/gold/soccer/`.

### Deliverables
- [x] `src/sports_ml/sports/soccer/features/team.py`: Rolling 5, 10, 19-match (half-season) and EWMA pre-match features:
  - **Underlying Shot & Chance Quality**: $npxGF/90$, $npxGA/90$, $npxGD/90$, $npxGF\%$, total $xGF/90$, $xGA/90$, shots per 90, and shots-on-target share ($SOT\%$).
  - **Tactical & Territorial Control**: Pressing intensity ($PPDA$), opponent $PPDA$ allowed, deep completions per 90, and corner/set-piece share.
  - **Finishing & Defensive Luck / Regression Signals**: $(GF - xGF)_{w10}$ and $(GA - xGA)_{w10}$ (measuring unsustainable finishing or goalkeeper variance analogous to NHL PDO).
  - **Schedule Congestion & Promoted Status**: Rest days, midweek turnaround flag (`rest_days <= 3`), matches in last 14 days, geodesic stadium travel distance, and promoted-club indicator (`is_promoted_team`).
- [x] `src/sports_ml/sports/soccer/features/goalkeeper.py`: Starting goalkeeper resolver and rolling Bayesian-shrunk Goals Saved Above Expected per 90 ($\text{GSAA}/90 = (\text{xGA}_{\text{faced}} - \text{GA})/90$), save percentage ($\text{SV\%}$), clean-sheet rate, and starter Elo modifier hook.
- [x] `src/sports_ml/sports/soccer/features/roster.py`: Active XI/squad aggregator weighting available outfield players' rolling per-90 $npxG$, $xA$, $xGChain$, and $xGBuildup$ by projected minutes share across positional units (`FWD`, `MID`, `DEF`), plus `missing_regular_minutes` from injured/suspended regulars.
- [x] `src/sports_ml/sports/soccer/features/player.py`: Stage 2 individual player-matchup feature builder (rolling minutes, start rate, availability weight from `chance_of_playing_next_round`, per-90 $npxG$, $xA$, shots, SOT, penalty/free-kick/corner taker flags, injury/suspension-vacated positional minutes, opponent defensive $xGA/90$ & $PPDA$, and opponent starting GK $\text{GSAA}/90$).

### Verification & Acceptance Criteria
- [x] `tests/sports/soccer/test_soccer_features.py`:
  - **Strict Anti-Leakage Test (Future-Mutation Invariance)**: Verify that mutating match goals, $xG$, cards, or player stats for matches at index $\ge t$ (including double-gameweek fixtures) produces **bit-for-bit identical** pre-match features at index $< t$.
  - **Cold-Start & Promoted-Club Test**: Verify Matchweek 1, newly promoted clubs, and debuting/transferred players produce finite, non-null Bayesian-shrunk features.
  - **Injury/Suspension Lineup Test**: Verify that marking a first-choice striker or penalty taker as suspended/injured (`chance_of_playing = 0`) updates `missing_regular_minutes` and redistributes projected minutes and set-piece weight cleanly.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 10: Stage 1 — Mandatory Calibrated 3-Way (`1X2`) & Goals Environment Pipeline
**Status**: `[x] Completed`

### Goal
Implement the mandatory Stage 1 EPL match outcome model producing calibrated 3-way probabilities $P(\text{Home Win})$, $P(\text{Draw})$, $P(\text{Away Win})$ and expected team goals $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ with Dixon-Coles low-score correlation adjustment and walk-forward backtesting.

### Deliverables
- [x] `src/sports_ml/sports/soccer/models/stage1_match.py`:
  - **Auxiliary Team-Goals Environment Model**: Poisson/Tweedie regressors estimating $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ paired with a **Dixon-Coles bivariate Poisson low-score correlation adjustment ($\rho$)** to derive structural scoreline probabilities $P(G_H = i, G_A = j)$ and implied Poisson `1X2` probabilities.
  - **Hybrid 3-Way Outcome Ensemble**: Blends the 3-way Elo baseline, Dixon-Coles goal-implied `1X2` probabilities, and a **Multiclass LightGBM Classifier (`objective="multiclass"`, `num_class=3`)** trained on pre-match team, tactical ($npxGD$, $PPDA$), schedule, goalkeeper, and lineup differentials.
  - **Out-of-Fold 3-Way Calibration**: Fits `MulticlassProbabilityCalibrator` strictly on held-out chronological validation folds so $P(\text{Home Win}) + P(\text{Draw}) + P(\text{Away Win}) = 1.0$ and $P_c \in (0, 1)$.
  - **Walk-Forward Backtesting Harness (`run_soccer_stage1_walk_forward_backtest`)**: Evaluates out-of-sample **Ranked Probability Score (RPS)**, 3-class **Log Loss**, multi-class **Brier Score**, and ECE against uniform (`1/3`), empirical EPL prior, and Elo-1X2 baselines, with `FoldBoundaryAudit` proving zero temporal overlap.

### Verification & Acceptance Criteria
- [x] `tests/sports/soccer/test_stage1_model.py`:
  - Verify predicted probabilities lie strictly in $(0, 1)$ and $P(\text{Home Win}) + P(\text{Draw}) + P(\text{Away Win}) = 1.0$ (within `1e-9`) for every match.
  - Verify walk-forward training never fits the GBDT, goal regressors, or 3-way calibrator on validation/test dates (`gbdt_train_max_time < calibrator_min_time <= calibrator_max_time < val_min_time`).
  - Verify calibrated model achieves lower Ranked Probability Score (RPS), Brier Score, and Log Loss than the empirical home/draw/away baseline on synthetic directional EPL data.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 11: Stage 2 Player Projections, Unified Soccer Predictor & CLI Integration
**Status**: `[x] Completed`

### Goal
Implement the optional Stage 2 EPL outfield & goalkeeper stat projection models conditioned on Stage 1, wire `SoccerMatchupPredictor` into the `sports-ml` CLI (`--sport soccer`), and update documentation and skills.

### Deliverables
- [x] `src/sports_ml/sports/soccer/models/stage2_player.py`:
  - **Outfield Minutes Projector**: Estimates expected minutes $\widehat{\text{Min}} \in [0, 90]$ for active/available squad players.
  - **Outfield Count/Rate Models**: LightGBM Poisson/Tweedie regressors conditioned on player per-90 rates, penalty/set-piece role, opponent defensive suppression, and Stage 1 context ($P(\text{Win})$, $P(\text{Draw})$, $\hat{\lambda}_{\text{team}}$) predicting `proj_goals`, `proj_assists`, `proj_goal_involvements` ($G+A$), `proj_shots`, `proj_sot`, `prob_1plus_goal`, `prob_1plus_assist`, and `prob_1plus_gi`.
  - **Goalkeeper Projection Model**: Predicts `proj_saves`, `proj_goals_against`, `proj_save_pct`, and `prob_clean_sheet` ($P(\text{GA} = 0)$ combining Stage 1 $\hat{\lambda}_{\text{opp}}$ Poisson clean-sheet probability $e^{-\hat{\lambda}_{\text{opp}}}$ with GK $\text{GSAA}/90$).
  - **Cross-Stage Coherence Reconciler (`reconcile_outfield_team_goals`)**: Soft-scales team outfield `proj_goals` and `proj_assists` toward Stage 1's $\hat{\lambda}_{\text{team}}$ ($\gamma = 0.50$, EPL assist-per-goal ratio $\approx 0.72$).
- [x] `src/sports_ml/sports/soccer/models/predictor.py`: `SoccerMatchupPredictor` supporting `predict_matchup(..., include_players: bool = False)` and `predict_date(..., include_players: bool = False)`, surfacing pre-match confidence warnings (unconfirmed starting XI vs. confirmed lineup, doubtful players with `chance_of_playing < 75%`, promoted-club early-season prior, Understat lag fallback), and appending predictions to the live audit ledger (`data/predictions/soccer_predictions.parquet`).
- [x] `src/sports_ml/cli.py`: Full multi-sport CLI routing for `--sport soccer` across `check-sources`, `ingest`, `features build`, `train`, `backtest`, `predict` (`--matchup AWAY@HOME`, `--include-players`, `--offline`), and `score-predictions` (scoring both binary NHL and 3-way `1X2` + RPS EPL ledgers), with Rich terminal tables for `1X2` match probabilities and positional player projections (`FWD`, `MID`, `DEF`, `GKP`).
- [x] `README.md` & `docs/architecture.md`: Update quickstart, CLI examples, and architecture diagrams to cover both `--sport hockey` (NHL) and `--sport soccer` (EPL).

### Verification & Acceptance Criteria
- [x] `tests/sports/soccer/test_stage2_player.py`:
  - Verify all predicted counts and minutes ($\in [0, 90]$) are non-negative ($\ge 0$) and milestone/clean-sheet probabilities lie in $[0, 1]$.
  - Verify **Stage-Conditioning Monotonicity**: Increasing Stage 1's expected team goals $\hat{\lambda}_{\text{team}}$ increases reconciled outfield goal and goal-involvement projections; increasing $\hat{\lambda}_{\text{opp}}$ decreases goalkeeper `prob_clean_sheet`.
  - Verify Stage 2 can be toggled completely off (`include_players=False`) with zero overhead when only Stage 1 `1X2` probabilities are requested.
- [x] `tests/sports/soccer/test_soccer_cli_e2e.py`: End-to-end offline CLI integration test running `ingest` $\rightarrow$ `features build` $\rightarrow$ `train` $\rightarrow$ `backtest` $\rightarrow$ `predict` (Stage 1 only & `--include-players`) $\rightarrow$ `score-predictions` for `--sport soccer`.
- [x] Full `pytest --cov=sports_ml`, `ruff check .`, `ruff format --check .`, and `mypy src tests` pass cleanly.

---

# Phase 3: Platform Scalability & Storage Optimization

## Milestone 12: Transparent Raw-Cache Compression, Deduplication & Storage Compaction (`core/cache.py` & `ingestion/`)
**Status**: `[ ] Planned`

### Goal
Shrink the local `data/raw/` footprint by **~93% (`6.2 GB` $\rightarrow$ `~415 MB`)** without losing a single historical season, game, or feature column by:
1. Deduplicating MoneyPuck's cumulative `all_teams.csv` cache into a single shared `teams_all.csv.gz` file instead of storing one `122 MB` copy per season (`teams_2000.csv`..`teams_2026.csv`).
2. Compressing all raw CSV caches as `.csv.gz` on ingestion and reading them directly via Polars (`pl.read_csv(path)` natively decompresses `.csv.gz` in Rust).
3. Compressing all raw JSON caches across NHL and EPL ingestion clients as `.json.gz` with transparent backward-compatible fallback to existing uncompressed `.csv` / `.json` files.
4. Providing a `sports-ml compact-cache` CLI command to migrate and compact existing local `data/raw/` directories in place with zero network calls.

### Deliverables
- [ ] `src/sports_ml/core/cache.py`: Sport-agnostic raw-cache storage & compaction helpers using Python's standard-library `gzip`:
  - `resolve_cache_read_path(preferred_gz_path: Path) -> Path | None`: Resolves `*.gz` first and falls back to the legacy uncompressed sibling path (`preferred_gz_path.with_suffix("")`) if present on disk.
  - `write_gzip_bytes(gz_path: Path, raw_bytes: bytes, *, compresslevel: int = 6) -> Path`: Atomically writes Gzip-compressed bytes (via a `.tmp` file rename) and removes any legacy uncompressed sibling file once written.
  - `read_cached_bytes(preferred_gz_path: Path) -> bytes`: Reads and decompresses `*.gz` or reads the legacy uncompressed file transparently.
  - `write_cached_json(gz_path: Path, payload: Any, *, compresslevel: int = 6) -> Path` and `read_cached_json(preferred_gz_path: Path, *, ttl_seconds: float | None = None) -> Any | None`: Compact JSON (`separators=(",", ":")`) Gzip read/write with optional `mtime` TTL check and legacy `.json` fallback.
  - `compact_raw_cache_directory(raw_dir: Path, *, dry_run: bool = False) -> CacheCompactionReport`: Scans `data/raw/` recursively:
    - Consolidates legacy `moneypuck/teams_*.csv` files into `moneypuck/teams_all.csv.gz` (preserving the newest file's payload and `.meta.json` sidecar) and removes the redundant per-season `teams_YYYY.csv` / `teams_YYYY.meta.json` copies.
    - Compresses all remaining uncompressed `.csv` $\rightarrow$ `.csv.gz` and `.json` (excluding `*.meta.json` ETag sidecars) $\rightarrow$ `.json.gz`, verifying readability before removing the uncompressed original.
    - Returns structured `CacheCompactionReport` (`files_compacted`, `duplicates_removed`, `bytes_before`, `bytes_after`, `bytes_saved`).
- [ ] `src/sports_ml/sports/hockey/ingestion/moneypuck.py`:
  - **Shared `teams_all.csv.gz` Cache Key**: When fetching `"teams"` from the default cumulative `all_teams.csv` endpoint (`self.base_url == MONEYPUCK_SEASON_URL.rstrip("/")`), use cache stem `teams_all` (`teams_all.csv.gz` + `teams_all.meta.json`) instead of `teams_{season_start_year}`. Refresh via TTL/ETag when `is_active_season=True` or when the requested `season_start_year` is not yet present in the cached file. (Preserve per-season `teams_{season_start_year}.csv.gz` when a custom per-season `base_url` is injected in tests).
  - **Compress-on-Ingest + Direct Polars Read**: Write extracted CSV bytes directly to `*.csv.gz` (`skaters_{YYYY}.csv.gz`, `goalies_{YYYY}.csv.gz`, `teams_all.csv.gz`) via `write_gzip_bytes`, and pass the resolved path (`*.csv.gz` or legacy `*.csv`) directly to `pl.read_csv(csv_path)`.
- [ ] `src/sports_ml/sports/hockey/ingestion/nhl_api.py`:
  - Update schedule, boxscore, landing, and roster disk caches (`schedules/*.json.gz`, `boxscores/{game_id}.json.gz`, `landings/{game_id}.json.gz`, `rosters/{team}_current.json.gz`) to write `.json.gz` and read `.json.gz` with automatic fallback to existing `.json` files via `core/cache.py`.
- [ ] `src/sports_ml/sports/soccer/ingestion/football_data.py`, `fpl_api.py`, and `understat.py`:
  - Update `FootballDataClient` (`E0_{code}.csv.gz`), `FPLAPIClient` (`bootstrap_static.json.gz`, `fixtures.json.gz`, `element_summaries/{id}.json.gz`), and `UnderstatClient` (`epl_teams_{yr}.json.gz`, `epl_players_{yr}.json.gz`) to compress on ingest and transparently read `.gz` or legacy uncompressed files.
- [ ] `src/sports_ml/sports/hockey/ingestion/reconciler.py` & `src/sports_ml/sports/soccer/ingestion/reconciler.py`:
  - Explicitly pass `compression="zstd"` to all `write_parquet(...)` calls in Silver and Gold table persistence.
- [ ] `src/sports_ml/cli.py`:
  - Add `sports-ml compact-cache [--data-dir data] [--dry-run]` CLI command printing a Rich summary table of files compacted, duplicate `teams_YYYY.csv` files removed, and disk space reclaimed (`bytes_before` $\rightarrow$ `bytes_after`).

### Verification & Acceptance Criteria
- [ ] `tests/core/test_cache_compaction.py`:
  - Verify `write_gzip_bytes` / `write_cached_json` and `pl.read_csv(gz_path)` round-trip both `.csv.gz` and `.json.gz` losslessly.
  - Verify backward-compatible fallback: pre-existing uncompressed `.csv` and `.json` files in a cache directory are read without network calls and compacted cleanly by `compact_raw_cache_directory`.
  - Verify `compact_raw_cache_directory` deduplicates multiple `teams_2023.csv`, `teams_2024.csv`, `teams_2025.csv` files into a single `teams_all.csv.gz` and supports `--dry-run` without mutating files.
- [ ] `tests/sports/hockey/test_ingestion.py` & `tests/sports/soccer/test_ingestion.py`:
  - Verify `MoneyPuckClient` downloads `all_teams.csv` once across multiple seasons into `teams_all.csv.gz`, stores `skaters_{YYYY}.csv.gz` and `goalies_{YYYY}.csv.gz`, and `pl.read_csv` parses identical Silver DataFrames.
  - Verify `NHLWebAPIClient`, `FootballDataClient`, `FPLAPIClient`, and `UnderstatClient` write `.gz` cache files on ingest and produce **bit-for-bit identical** Silver Parquet tables.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.


