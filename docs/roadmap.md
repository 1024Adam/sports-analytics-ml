# Phased Implementation Roadmap

> **Related Documents**: [System Architecture Blueprint](./architecture.md) | [Data & Feature Specification](./data-and-features.md)

This roadmap breaks the `sports-analytics-ml` platform into **6 sequential, testable milestones** designed for execution by the `sports-ml-developer` agent. Each milestone has explicit file deliverables, acceptance criteria, and mandatory `pytest` anti-leakage gates.

---

## Milestone 1: Project Foundation & Sport-Agnostic Core Engine (`core/`)
**Status**: `[ ] Not Started`

### Goal
Establish the Python 3.11+ package structure (`pyproject.toml`, `src/sports_ml/`), canonical data contracts, and the sport-agnostic time-series and rating primitives in `src/sports_ml/core/`.

### Deliverables
- [ ] `.gitignore`: Strictly excludes `data/`, `artifacts/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, and `.pytest_cache/`.
- [ ] `pyproject.toml` and `uv.lock` configured with `polars`, `pyarrow`, `pydantic`, `httpx`, `tenacity`, `lightgbm`, `scikit-learn`, `scipy`, `typer`, `rich`, `pytest`, `pytest-cov`, `mypy`, and `ruff` (with rule sets `E`, `F`, `I`, `B`, `UP`, `SIM`, `NPY`), plus `src/sports_ml/config.py` defining deterministic `RANDOM_SEED = 42`.
- [ ] `.github/workflows/ci.yml`: Fast, offline GitHub Actions CI workflow triggered on `push` and `pull_request` to `main` (with `permissions: contents: read`) running a tracked-binary/data guard check, `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml`.
- [ ] `src/sports_ml/core/contracts.py`: `SportAdapter` protocol supporting both `head_to_head` (NHL, MLB, PGA H2H) and `multi_entrant_field` (PGA tournament field) competition formats, canonical event/participant schema validators, and starter/venue modifier hooks.
- [ ] `src/sports_ml/core/temporal.py`: Leak-free Polars window helpers (`pregame_rolling_mean`, `pregame_rolling_sum`, `pregame_ewma`, and generalized `pregame_exposure_rate` supporting per-60 TOI, per-PA/per-9 IP, and per-round Strokes Gained) enforcing `.shift(1)` ordered by `(start_time_utc, event_seq, event_id)`.
- [ ] `src/sports_ml/core/shrinkage.py`: Early-season / low-sample Bayesian prior blending (`(n * curr + k * prior) / (n + k)`).
- [ ] `src/sports_ml/core/ratings.py`: Dynamic rating engine supporting pairwise Elo/Glicko updates (with Home-Field Advantage, Margin-of-Victory multiplier, OT/Shootout dampening, starter adjustment hooks, and between-season regression) and extensible multi-entrant field skill ratings.
- [ ] `src/sports_ml/core/schedule.py`: Rest days, back-to-back (`is_b2b`), same-day doubleheader turnaround (`event_seq`), `k`-in-`m` nights density, geodesic venue haversine distance, and timezone shift calculator.
- [ ] `src/sports_ml/core/validation.py`: Walk-forward expanding and sliding window time-series cross-validator grouped by chronological event timestamp.

### Verification & Acceptance Criteria
- [ ] `tests/core/test_temporal_leakage.py`: **Future-Mutation Invariance Test** — mutating all target and box-score columns for events at index $\ge t$ (including same-day doubleheader Game 2 or multi-round tournament Round $r+1$) produces **bit-for-bit identical** features at index $< t$.
- [ ] `tests/core/test_ratings.py`: Verify Elo zero-sum conservation, home-field shift, shootout dampening, and off-season mean regression.
- [ ] `tests/core/test_schedule.py`: Verify B2B, same-day doubleheader, and 3-in-4 calculations across month/year boundaries and venue travel distances.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 2: Hybrid NHL Ingestion & Silver Lakehouse (`sports/hockey/ingestion/`)
**Status**: `[ ] Not Started`

### Goal
Build resilient, cached ingestion pipelines for the **Official NHL Web API** and **MoneyPuck CSV datasets**, normalizing and reconciling them into canonical Silver Parquet tables.

### Deliverables
- [ ] `src/sports_ml/sports/hockey/constants.py`: NHL 32-team registry, historical tricode normalization (`ARI` $\rightarrow$ `UTA`, `S.J` $\rightarrow$ `SJS`, `L.A` $\rightarrow$ `LAK`, `N.J` $\rightarrow$ `NJD`, `T.B` $\rightarrow$ `TBL`), and arena latitude/longitude/timezone coordinates.
- [ ] `src/sports_ml/sports/hockey/ingestion/nhl_api.py`: `httpx` client with deterministic disk caching (`data/raw/nhl_api/`) and `tenacity` retry/backoff for schedules, game boxscores, and team rosters. Supports permanent caching for `FINAL`/`OFF` games and incremental delta fetching (`--current-season` / `--start-date` / `--end-date`) for newly completed games and upcoming `FUT` schedules.
- [ ] `src/sports_ml/sports/hockey/ingestion/moneypuck.py`: Downloader and Polars parser for MoneyPuck team, skater, and goalie game-by-game CSVs (`5on5`, `5on4`, `4on5`, `all`), with TTL/ETag refresh for current-season-to-date CSV snapshots.
- [ ] `src/sports_ml/sports/hockey/ingestion/lineups.py`: **2-Tier Live Pre-Game Lineup, Starting Goalie & Injury Resolver**:
  - **Tier 1**: Official NHL API pre-game/warmup dressed lineup & starter (`/v1/gamecenter/{GAME_ID}/boxscore` + `/v1/roster/{TEAM}/current`).
  - **Tier 2**: Daytime morning-skate starting goalies (`Confirmed` / `Expected`), projected line combinations, and injury exclusions (e.g., DailyFaceoff feed reconciled to NHL 7-digit `player_id`s) when predicting hours before puck drop.
- [ ] `src/sports_ml/sports/hockey/ingestion/reconciler.py`: Upserts and joins NHL API and MoneyPuck records into the 4 canonical Silver Parquet tables (`games.parquet`, `team_game_logs.parquet`, `skater_game_logs.parquet`, `goalie_game_logs.parquet`), with graceful fallback when MoneyPuck lags by a day.

### Verification & Acceptance Criteria
- [ ] `tests/sports/hockey/test_ingestion.py`: Test JSON/CSV parsing, tricode normalization, schema adherence, incremental upsert (transitioning a game from `FUT` to `FINAL` as the current season progresses), 2-Tier pre-game lineup/goalie/injury resolution (Tier 1 warmup payload vs. Tier 2 daytime morning-skate payload), and fallback handling using deterministic local fixtures.
- [ ] Verify Silver tables enforce uniqueness on `(game_id)` for games, `(game_id, team)` for team logs, and `(game_id, player_id)` for player logs.

---

## Milestone 3: Hockey Pre-Game Feature Engineering (`sports/hockey/features/`)
**Status**: `[ ] Not Started`

### Goal
Construct the complete pre-game feature suite for **Stage 1 (Game Outcome)** and **Stage 2 (Player Matchup Projections)** in `data/gold/hockey/`.

### Deliverables
- [ ] `src/sports_ml/sports/hockey/features/team.py`: Rolling 5, 10, 25-game and EWMA features for 5v5 $xGF\%$, $xGF/60$, $xGA/60$, Corsi ($CF\%$), Fenwick ($FF\%$), High-Danger $xG\%$, PDO luck/regression signal, and Special Teams Net Edge (PP vs. PK).
- [ ] `src/sports_ml/sports/hockey/features/goalie.py`: Starting goalie resolver and rolling shrunk $\text{GSAx/60}$, $\text{SV\%}$, and goalie workload/fatigue metrics.
- [ ] `src/sports_ml/sports/hockey/features/roster.py`: Active roster aggregator weighting individual skater rolling $\text{ixG/60}$, $\text{OnIce-xGD/60}$, and $\text{GameScore/60}$ by projected TOI share for Forwards and Defensemen, plus `missing_regular_toi`.
- [ ] `src/sports_ml/sports/hockey/features/player.py`: Stage 2 individual player-matchup feature builder (rolling EV/PP TOI, vacated injury TOI, per-60 rates, shrunk career shooting %, opponent defensive suppression, and opponent starting goalie GSAx).

### Verification & Acceptance Criteria
- [ ] `tests/sports/hockey/test_hockey_features.py`:
  - **Strict Anti-Leakage Test**: Verify that changing tonight's game goals, $xG$, or player points does not alter any pre-game team, goalie, roster, or player feature for tonight's game.
  - **Cold-Start Test**: Verify Game 1 of a season and a debuting rookie/goalie produce finite, non-null shrunk features.
  - **Roster Weighting Test**: Verify TOI weights normalize cleanly and reflect lineup changes when a top-line player is removed from the active roster.

---

## Milestone 4: Stage 1 — Mandatory Calibrated Win Probability Pipeline
**Status**: `[ ] Not Started`

### Goal
Implement the mandatory Stage 1 game outcome model, probability calibration wrapper, auxiliary team-goals environment estimator, and walk-forward backtesting harness.

### Deliverables
- [ ] `src/sports_ml/core/calibration.py`: Out-of-fold Platt (Sigmoid) and Isotonic probability calibrators.
- [ ] `src/sports_ml/core/evaluation.py`: Evaluation suite computing **Log Loss**, **Brier Score**, Expected Calibration Error (ECE), accuracy, and benchmark comparisons against coin-flip, home-team baseline, and Elo-only baseline.
- [ ] `src/sports_ml/sports/hockey/models/stage1_game.py`:
  - Trained Elo baseline + LightGBM classifier producing calibrated $P(\text{Home Win})$ and $P(\text{Away Win}) = 1 - P(\text{Home Win})$.
  - Auxiliary Poisson/Tweedie team-goals regressor estimating $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ to provide macro pace context for Stage 2.

### Verification & Acceptance Criteria
- [ ] `tests/sports/hockey/test_stage1_model.py`:
  - Verify predicted probabilities lie strictly in $(0, 1)$ and $P(\text{Home Win}) + P(\text{Away Win}) = 1.0$.
  - Verify walk-forward training never fits the GBDT or calibrator on validation/test dates.
  - Verify calibrated model achieves lower Brier Score / Log Loss than a constant home-win-rate baseline on synthetic directional data.

---

## Milestone 5: Stage 2 — Optional Player Matchup Stat Projections
**Status**: `[ ] Not Started`

### Goal
Implement the optional Stage 2 player stat projection models for both teams' skaters and goalies, conditioned on Stage 1's game outcome and scoring environment.

### Deliverables
- [ ] `src/sports_ml/sports/hockey/models/stage2_player.py`:
  - **Skater TOI Projector**: Estimates $\widehat{\text{TOI}}_{\text{ev}}$ and $\widehat{\text{TOI}}_{\text{pp}}$ for active roster skaters.
  - **Skater Count/Rate Models**: LightGBM Poisson/Tweedie regressors conditioned on player rolling rates, opponent defense/goalie features, and Stage 1 context ($P(\text{Win})$, $\hat{\lambda}_{\text{team}}$) predicting `proj_goals`, `proj_assists`, `proj_points`, `proj_sog`, `prob_1plus_goal`, and `prob_1plus_point`.
  - **Goalie Projection Model**: Predicts `proj_shots_against`, `proj_goals_against`, `proj_saves`, and `proj_save_pct`.
  - **Cross-Stage Coherence Reconciler**: Soft-scales team skater goal totals toward Stage 1's $\hat{\lambda}_{\text{team}}$.

### Verification & Acceptance Criteria
- [ ] `tests/sports/hockey/test_stage2_player.py`:
  - Verify all predicted counts and TOI are non-negative ($\ge 0$) and milestone probabilities lie in $[0, 1]$.
  - Verify **Stage-Conditioning Monotonicity**: Increasing Stage 1's expected team goals $\hat{\lambda}_{\text{team}}$ or facing a weaker opponent goalie (lower GSAx) increases reconciled skater goal/point projections.
  - Verify Stage 2 can be toggled completely off (`include_players=False`) with zero overhead when only Stage 1 win probability is requested.

---

## Milestone 6: Unified Two-Stage Predictor Service & CLI (`predictor.py` & `cli.py`)
**Status**: `[ ] Not Started`

### Goal
Wire Stage 1 and Stage 2 into a cohesive inference API, artifact registry, and `sports-ml` CLI tool with Rich terminal output.

### Deliverables
- [ ] `src/sports_ml/core/registry.py`: Model bundle save/load with `metadata.json` fingerprinting (ordered feature names, Polars dtypes, SHA-256 schema hash, training date range, git SHA, and validation metrics) that validates feature compatibility before inference.
- [ ] `src/sports_ml/sports/hockey/models/predictor.py`: `HockeyMatchupPredictor` supporting `predict_matchup(..., include_players: bool = False)` and `predict_date(..., include_players: bool = False)`, surfacing pre-prediction confidence warnings (confirmed vs. projected starting goalie, data freshness, rookie prior flags), and automatically appending predictions to the live audit ledger (`data/predictions/hockey_predictions.parquet`).
- [ ] `src/sports_ml/cli.py`: `typer` CLI implementing:
  - Streamlined auto-chaining: `train` automatically builds/updates Gold features if missing or stale; `predict` connects live by default (with `--offline` opt-out) to run incremental `--current-season` ingestion, resolve live 2-Tier pre-game lineups/goalies/injuries, and refresh features before predicting.
  - Granular developer & audit subcommands: `ingest`, `features build`, `backtest` (R&D evaluation), `train`, `predict` (with `--include-players`, `--matchup`, and `--offline` flags), and `score-predictions` (evaluates the live `data/predictions/` ledger against completed games).
- [ ] `README.md`: End-to-end quickstart guide, CLI usage examples, and architecture overview.

### Verification & Acceptance Criteria
- [ ] Verify `registry.py` raises a clear schema-mismatch error if a feature column is missing or reordered at inference time.
- [ ] End-to-end CLI integration test running `ingest` $\rightarrow$ `features` $\rightarrow$ `train` $\rightarrow$ `backtest` $\rightarrow$ `predict` $\rightarrow$ `score-predictions` (both default Stage 1 only and Stage 1 + Stage 2 with `--include-players`) on fixture data.
- [ ] Full `pytest --cov=sports_ml`, `ruff check .`, `ruff format --check .`, and `mypy src tests` pass cleanly.
