# Phased Implementation Roadmap

> **Related Documents**: [System Architecture Blueprint](./architecture.md) | [NHL Data & Feature Specification](./hockey-data-and-features.md) | [EPL Soccer Data & Feature Specification](./soccer-data-and-features.md) | [MLB & NPB Baseball Data & Feature Specification](./baseball-data-and-features.md) | Football (NFL) Data & Feature Specification (`./football-data-and-features.md`, authored in Milestone 22)

This roadmap breaks the `sports-analytics-ml` platform into sequential, testable milestones across seven phases (**Phase 1: Core Engine & NHL Hockey**, Milestones 1–6; **Phase 2: English Premier League (EPL) Soccer**, Milestones 7–11; **Phase 3: Platform Scalability & Storage Optimization**, Milestone 12; **Phase 4: Market Benchmarking, Cross-Competition Fatigue & Micro-Features**, Milestones 13–15; **Phase 5: Dual-League MLB & NPB Baseball Plugin**, Milestones 16–21; **Phase 6: Football (NFL) Plugin**, Milestones 22–29; and **Phase 7: User Personalization**, Milestone 30) designed for execution by the `sports-ml-developer` agent. Each milestone has explicit file deliverables, acceptance criteria, and mandatory `pytest` anti-leakage gates.

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
**Status**: `[x] Completed`

### Goal
Shrink the local `data/raw/` footprint by **~93% (`6.2 GB` $\rightarrow$ `~415 MB`)** without losing a single historical season, game, or feature column by:
1. Deduplicating MoneyPuck's cumulative `all_teams.csv` cache into a single shared `teams_all.csv.gz` file instead of storing one `122 MB` copy per season (`teams_2000.csv`..`teams_2026.csv`).
2. Compressing all raw CSV caches as `.csv.gz` on ingestion and reading them directly via Polars (`pl.read_csv(path)` natively decompresses `.csv.gz` in Rust).
3. Compressing all raw JSON caches across NHL and EPL ingestion clients as `.json.gz` with transparent backward-compatible fallback to existing uncompressed `.csv` / `.json` files.
4. Providing a `sports-ml compact-cache` CLI command to migrate and compact existing local `data/raw/` directories in place with zero network calls.

### Deliverables
- [x] `src/sports_ml/core/cache.py`: Sport-agnostic raw-cache storage & compaction helpers using Python's standard-library `gzip`:
  - `resolve_cache_read_path(preferred_gz_path: Path) -> Path | None`: Resolves `*.gz` first and falls back to the legacy uncompressed sibling path (`preferred_gz_path.with_suffix("")`) if present on disk.
  - `write_gzip_bytes(gz_path: Path, raw_bytes: bytes, *, compresslevel: int = 6) -> Path`: Atomically writes Gzip-compressed bytes (via a `.tmp` file rename) and removes any legacy uncompressed sibling file once written.
  - `read_cached_bytes(preferred_gz_path: Path) -> bytes`: Reads and decompresses `*.gz` or reads the legacy uncompressed file transparently.
  - `write_cached_json(gz_path: Path, payload: Any, *, compresslevel: int = 6) -> Path` and `read_cached_json(preferred_gz_path: Path, *, ttl_seconds: float | None = None) -> Any | None`: Compact JSON (`separators=(",", ":")`) Gzip read/write with optional `mtime` TTL check and legacy `.json` fallback.
  - `compact_raw_cache_directory(raw_dir: Path, *, dry_run: bool = False) -> CacheCompactionReport`: Scans `data/raw/` recursively:
    - Consolidates legacy `moneypuck/teams_*.csv` files into `moneypuck/teams_all.csv.gz` (preserving the newest file's payload and `.meta.json` sidecar) and removes the redundant per-season `teams_YYYY.csv` / `teams_YYYY.meta.json` copies.
    - Compresses all remaining uncompressed `.csv` $\rightarrow$ `.csv.gz` and `.json` (excluding `*.meta.json` ETag sidecars) $\rightarrow$ `.json.gz`, verifying readability before removing the uncompressed original.
    - Returns structured `CacheCompactionReport` (`files_compacted`, `duplicates_removed`, `bytes_before`, `bytes_after`, `bytes_saved`).
- [x] `src/sports_ml/sports/hockey/ingestion/moneypuck.py`:
  - **Shared `teams_all.csv.gz` Cache Key**: When fetching `"teams"` from the default cumulative `all_teams.csv` endpoint (`self.base_url == MONEYPUCK_SEASON_URL.rstrip("/")`), use cache stem `teams_all` (`teams_all.csv.gz` + `teams_all.meta.json`) instead of `teams_{season_start_year}`. Refresh via TTL/ETag when `is_active_season=True` or when the requested `season_start_year` is not yet present in the cached file. (Preserve per-season `teams_{season_start_year}.csv.gz` when a custom per-season `base_url` is injected in tests).
  - **Compress-on-Ingest + Direct Polars Read**: Write extracted CSV bytes directly to `*.csv.gz` (`skaters_{YYYY}.csv.gz`, `goalies_{YYYY}.csv.gz`, `teams_all.csv.gz`) via `write_gzip_bytes`, and pass the resolved path (`*.csv.gz` or legacy `*.csv`) directly to `pl.read_csv(csv_path)`.
- [x] `src/sports_ml/sports/hockey/ingestion/nhl_api.py`:
  - Update schedule, boxscore, landing, and roster disk caches (`schedules/*.json.gz`, `boxscores/{game_id}.json.gz`, `landings/{game_id}.json.gz`, `rosters/{team}_current.json.gz`) to write `.json.gz` and read `.json.gz` with automatic fallback to existing `.json` files via `core/cache.py`.
- [x] `src/sports_ml/sports/soccer/ingestion/football_data.py`, `fpl_api.py`, and `understat.py`:
  - Update `FootballDataClient` (`E0_{code}.csv.gz`), `FPLAPIClient` (`bootstrap_static.json.gz`, `fixtures.json.gz`, `element_summaries/{id}.json.gz`), and `UnderstatClient` (`epl_teams_{yr}.json.gz`, `epl_players_{yr}.json.gz`) to compress on ingest and transparently read `.gz` or legacy uncompressed files.
- [x] `src/sports_ml/sports/hockey/ingestion/reconciler.py` & `src/sports_ml/sports/soccer/ingestion/reconciler.py`:
  - Explicitly pass `compression="zstd"` to all `write_parquet(...)` calls in Silver and Gold table persistence.
- [x] `src/sports_ml/cli.py`:
  - Add `sports-ml compact-cache [--data-dir data] [--dry-run]` CLI command printing a Rich summary table of files compacted, duplicate `teams_YYYY.csv` files removed, and disk space reclaimed (`bytes_before` $\rightarrow$ `bytes_after`).

### Verification & Acceptance Criteria
- [x] `tests/core/test_cache_compaction.py`:
  - Verify `write_gzip_bytes` / `write_cached_json` and `pl.read_csv(gz_path)` round-trip both `.csv.gz` and `.json.gz` losslessly.
  - Verify backward-compatible fallback: pre-existing uncompressed `.csv` and `.json` files in a cache directory are read without network calls and compacted cleanly by `compact_raw_cache_directory`.
  - Verify `compact_raw_cache_directory` deduplicates multiple `teams_2023.csv`, `teams_2024.csv`, `teams_2025.csv` files into a single `teams_all.csv.gz` and supports `--dry-run` without mutating files.
- [x] `tests/sports/hockey/test_ingestion.py` & `tests/sports/soccer/test_ingestion.py`:
  - Verify `MoneyPuckClient` downloads `all_teams.csv` once across multiple seasons into `teams_all.csv.gz`, stores `skaters_{YYYY}.csv.gz` and `goalies_{YYYY}.csv.gz`, and `pl.read_csv` parses identical Silver DataFrames.
  - Verify `NHLWebAPIClient`, `FootballDataClient`, `FPLAPIClient`, and `UnderstatClient` write `.gz` cache files on ingest and produce **bit-for-bit identical** Silver Parquet tables.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

# Phase 4: Market Benchmarking, Cross-Competition Fatigue & Micro-Event Play-by-Play

## Milestone 13: Evaluation-Only Historical Closing Odds, De-Vigging & CLV Backtest Benchmarking
**Status**: `[x] Completed`

### Goal
Benchmark fundamental Stage 1 predictions against sharp sportsbook closing lines (Pinnacle / Bet365 / Market Consensus) in `sports-ml backtest` and `sports-ml score-predictions` while enforcing a **strict anti-contamination firewall** that forbids betting odds from ever entering `FEATURE_COLUMNS` (`X`).

### Architectural Firewall (Zero Odds Contamination)
1. **Evaluation-Only Namespace (`market_*`)**: All odds and market-implied probability columns are prefixed with `market_` (`market_home_decimal_odds`, `market_draw_decimal_odds`, `market_away_decimal_odds`, `market_prob_home`, `market_prob_draw`, `market_prob_away`, `market_overround`).
2. **Hard Feature Firewall**: Both `sports/hockey/features/pipeline.py` and `sports/soccer/features/pipeline.py` (plus a dedicated anti-contamination test) explicitly strip/ban any `market_*` or `*odds*` columns from `STAGE1_FEATURE_COLUMNS` and `STAGE2_FEATURE_COLUMNS` so the ML models remain **100% first-principles fundamental models**.

### Deliverables
- [x] `src/sports_ml/core/market.py`: Sport-agnostic market odds math, de-vigging, and edge/ROI evaluation:
  - `american_to_decimal(odds: float | NDArray) -> float | NDArray` and `decimal_to_raw_prob(decimal_odds: float | NDArray) -> float | NDArray`.
  - `devig_probabilities(decimal_odds: NDArray[np.float64], method: Literal["power", "multiplicative", "shin"] = "power") -> NDArray[np.float64]`:
    - **Multiplicative (Proportional)**: $P_i = \frac{1/d_i}{\sum_j 1/d_j}$.
    - **Power Method (Default for 2-way NHL & 3-way EPL)**: Solves for exponent $k > 1$ such that $\sum_i (1/d_i)^k = 1.0$, accounting for the favorite-longshot bias.
  - `evaluate_market_edge(...) -> MarketEdgeReport`: Computes head-to-head metrics on the subset of backtest games with closing odds:
    - Model vs. Market **Log Loss**, **Brier Score**, **RPS** (soccer), and **ECE**.
    - **Disagreement / Residual Slices**: Model accuracy, Brier score, and simulated flat-stake & fractional-Kelly ROI when $|P_{\text{model}} - P_{\text{market}}| \ge \tau$ (for $\tau \in \{0.03, 0.05, 0.08\}$), isolating situational pockets where the fundamental model beats the closing line.
- [x] `src/sports_ml/sports/soccer/ingestion/football_data.py` & `reconciler.py`:
  - Extract historical closing & opening `1X2` and Over/Under `2.5` odds already embedded in the cached `Football-Data.co.uk` `E0_{code}.csv.gz` files (`PSCH`/`PSCD`/`PSCA` Pinnacle closing, falling back to `B365CH`/`B365CD`/`B365CA` or opening `PSH`/`PSD`/`PSA` / `B365H`/`B365D`/`B365A` for earlier seasons) with zero extra network calls.
  - Persist `market_home_decimal_odds`, `market_draw_decimal_odds`, `market_away_decimal_odds`, and power-de-vigged `market_prob_home`, `market_prob_draw`, `market_prob_away` as nullable evaluation columns in `data/silver/soccer/matches.parquet`.
- [x] `src/sports_ml/sports/hockey/ingestion/odds.py` & `reconciler.py`:
  - `NHLOddsIngestor`: Supports importing local/public historical NHL closing moneyline & total CSV archives (`data/raw/odds/nhl_historical_odds.csv.gz`, converting American moneylines like `-145` / `+125` to decimal and power-de-vigged probabilities) plus an optional pre-game odds snapshot hook (`THE_ODDS_API_KEY` env var if provided, gracefully skipped when absent).
  - Persist nullable `market_home_decimal_odds`, `market_away_decimal_odds`, `market_prob_home`, `market_prob_away` on `data/silver/hockey/games.parquet`.
- [x] `src/sports_ml/core/evaluation.py` & `src/sports_ml/cli.py`:
  - Extend `BenchmarkComparisonReport` and `MulticlassBenchmarkReport` with optional `market_closing_baseline` and `MarketEdgeReport`.
  - Update `sports-ml backtest` and `sports-ml score-predictions` Rich CLI tables to display the **De-Vigged Market Closing Line** benchmark alongside Elo/prior baselines and print the **Fundamental vs. Market Disagreement (`>5% Edge`)** performance breakdown when odds are present.

### Verification & Acceptance Criteria
- [x] `tests/core/test_market.py`:
  - Verify `devig_probabilities` (`"multiplicative"` and `"power"`) produces probabilities strictly in $(0, 1)$ that sum to `1.0` (within `1e-9`) for both 2-way NHL moneylines and 3-way EPL `1X2` odds, and that the `"power"` method shrinks longshot probabilities more than heavy favorites.
  - Verify `evaluate_market_edge` accurately computes flat-stake ROI, Kelly edge, and disagreement-bucket accuracy.
- [x] `tests/sports/hockey/test_odds_firewall.py` & `tests/sports/soccer/test_odds_firewall.py`:
  - **Strict Anti-Contamination Test**: Verify that no column starting with `market_` or containing `odds` is ever present in `STAGE1_FEATURE_COLUMNS` or `STAGE2_FEATURE_COLUMNS`, and that mutating `market_*` columns in Silver tables changes `0.0` of the model's predicted probabilities while updating the backtest market benchmark report.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 14: EPL Cross-Competition Schedule & Squad Fatigue (UEFA Champions/Europa/Conference League, FA Cup & EFL Cup)
**Status**: `[x] Completed`

### Goal
Eliminate the EPL-only schedule blind spot by ingesting non-league competitive fixtures (**UEFA Champions League**, **Europa League**, **Conference League**, **FA Cup**, and **EFL Carabao Cup**) for all 20 EPL clubs and using them **strictly to compute true multi-competition schedule rest, European travel distance, 14-day match congestion, and squad rotation/minutes load**—without polluting domestic EPL tactical $xG$ rates or Elo ratings with lower-tier cup opponents.

### Deliverables
- [x] `src/sports_ml/sports/soccer/ingestion/non_league_fixtures.py` & `reconciler.py`:
  - Ingest historical and active-season non-EPL club fixtures (`competition_type` $\in \{\text{"UCL"}, \text{"UEL"}, \text{"UECL"}, \text{"FA\_CUP"}, \text{"EFL\_CUP"}\}$) from Understat/Football-Data/FPL cup & European feeds into `data/silver/soccer/club_all_competitions_schedule.parquet` (`match_date`, `start_time_utc`, `team`, `opponent`, `competition_type`, `is_home`, `venue_lat`, `venue_lon`, `extra_time_played`, `squad_minutes_load`).
- [x] `src/sports_ml/sports/soccer/features/team.py` & `roster.py`:
  - Upgrade pre-match EPL schedule & fatigue features to query the unified all-competition timeline (`strictly < T_match`) before every EPL fixture:
    - `true_all_comp_rest_days`: Calendar/hour rest since the club's last competitive match in **any** competition (correctly identifying a Wednesday Champions League match before a Saturday EPL kickoff as `2.5–3 days` rest instead of `7 days`).
    - `had_midweek_european_match`: Binary indicator (`1` if the club played a `UCL` / `UEL` / `UECL` fixture within the last 5 days).
    - `had_midweek_away_european_match` & `european_travel_km_last_7d`: Captures long-haul midweek European away trips (e.g., Champions League away match followed by an early Saturday EPL kickoff).
    - `all_comp_matches_last_14d` & `all_comp_matches_last_28d`: True multi-competition fixture density.
    - `cup_extra_time_last_7d`: Flag for 120-minute FA Cup / EFL Cup / UEFA knockout ties in the prior 7 days.
    - `active_xi_14d_all_comp_minutes`: Rolling 14-day cumulative minutes across all competitions for projected starters (`roster.py`).
  - **Tactical Isolation Guarantee**: Ensure rolling $npxGF/90$, $npxGA/90$, $PPDA$, and EPL Elo updates continue to filter exclusively on `competition_type == "EPL"` so facing a League One club in the FA Cup never distorts a club's Premier League shot-quality ratings.

### Verification & Acceptance Criteria
- [x] `tests/sports/soccer/test_cross_competition_fatigue.py`:
  - Verify that inserting a Wednesday UEFA Champions League away match between two Saturday EPL fixtures reduces the Saturday EPL pre-match `true_all_comp_rest_days` from `7.0` to `3.0`, sets `had_midweek_european_match = 1`, and increases `all_comp_matches_last_14d` by `+1`.
  - **Tactical Isolation Test**: Verify that mutating goals or $xG$ in `UCL` / `FA_CUP` / `EFL_CUP` fixtures produces **bit-for-bit identical** EPL Elo ratings and rolling EPL $npxGF/90$ & $PPDA$ features.
  - **Temporal Firewall Test**: Verify that future UEFA/Cup fixtures at $\ge T_{\text{match}}$ never leak into pre-match features at $< T_{\text{match}}$.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 15: Shot-Coordinate & Shift-Level Play-by-Play Micro-Features (Score-Adjusted $xG$, Rush/Rebound Creation & TOI Quality of Competition)
**Status**: `[x] Completed`

### Goal
Move beyond coarse game-level boxscore aggregates by ingesting **shot-level event streams** and **shift/matchup summaries** for both NHL Hockey and EPL Soccer, compressing millions of micro-events into compact, high-signal pre-game Silver/Gold features (**Score-Adjusted $xG$**, **Rush & Rebound Shot Creation**, **Slot Shot Share**, **TOI Quality of Competition**, and **Open-Play vs. Set-Piece $xG$ Split**).

### Storage & Compaction Design
Instead of keeping 15+ GB of raw per-game JSONs uncompressed on disk, the ingestion layer streams bulk shot archives (`MoneyPuck` shot zips for NHL, `Understat` match shot arrays for EPL) and NHL shift/play-by-play summaries, aggregates them into compact **game-team** and **game-player micro-feature Silver tables** (`zstd`-compressed Parquet, adding only `~50–120 MB` total to `data/silver/`), and caches raw archives as `.csv.gz` / `.json.gz`.

### Deliverables
- [x] `src/sports_ml/sports/hockey/ingestion/pbp_shots.py` & `shifts.py`:
  - Ingest MoneyPuck bulk shot-level archives (`shots_{YYYY}.csv.gz`) and NHL API play-by-play / shift-chart summaries (`/v1/gamecenter/{GAME_ID}/play-by-play`) into `data/silver/hockey/team_pbp_micro_logs.parquet` and `skater_pbp_micro_logs.parquet`:
    - **Score-Venue Adjusted 5v5 $xG$ & Fenwick (`score_adj_xgf_5v5`, `score_adj_xga_5v5`, `score_adj_ff_5v5`)**: Discounts trailing-team "score effects" (garbage-time shot volume when down by $2+$ goals) and boosts shots generated while tied or leading.
    - **Rush & Rebound $xG$ Generation (`rush_xgf_5v5`, `rebound_xgf_5v5`, `rebound_xga_5v5`)**: Separates transition offense and net-front second-chance creation from perimeter point shots, and measures goalie rebound control (`rebound_xga_per_60`).
    - **Shot Distance, Angle & Slot Share (`avg_shot_distance_5v5`, `inner_slot_xg_share`)**: Spatial shot-profile quality derived from $(x, y)$ ice coordinates.
    - **Skater TOI Quality of Competition (`qoc_opponent_toi_share`) & Quality of Teammates (`qot_teammate_ixg_per_60`)**: Measures whether a forward line or defense pair plays heavy shutdown minutes against the opponent's top-6 forwards or sheltered 3rd-pair matchups.
- [x] `src/sports_ml/sports/soccer/ingestion/shot_events.py`:
  - Extract shot-level event records (`situation` $\in \{\text{"OpenPlay"}, \text{"FromCorner"}, \text{"SetPiece"}, \text{"DirectFreekick"}, \text{"Penalty"}\}$, `shotType`, $(X, Y)$ pitch coordinates, `lastAction` $\in \{\text{"Throughball"}, \text{"Cross"}, \text{"Rebound"}, \text{"Standard"}\}$, and minute/score-state at shot timestamp) from Understat match payloads into `data/silver/soccer/team_shot_micro_logs.parquet`:
    - **Open-Play vs. Set-Piece $xG$ Split (`open_play_xgf`, `open_play_xga`, `set_piece_xgf`, `set_piece_xga`)**: Isolates repeatable open-play chance creation from set-piece specialists and corners/free-kicks matchup edges.
    - **Game-State Neutral / Tied $npxG$ Rate (`tied_state_npxgd_per_90`)**: Filters out late-game desperation shots when a team trails by $2+$ goals vs. a low block.
    - **Box-Entry & Throughball Danger (`box_shot_share`, `throughball_xgf`)**: Spatial shot quality inside the 18-yard box and 6-yard zone.
- [x] `src/sports_ml/sports/hockey/features/` & `src/sports_ml/sports/soccer/features/`:
  - Wire rolling left-closed (`.shift(1)`) micro-features into Stage 1 team matchup differentials (e.g., `diff_score_adj_xgf_pct_5v5`, `diff_rush_rebound_xg_rate`, `diff_set_piece_xg_edge`, `diff_open_play_npxgd_90`) and Stage 2 player projections (skater `rolling_rush_rebound_ixg_per_60`, `rolling_qoc_index`; outfield `rolling_open_play_npxg_per_90`, `rolling_set_piece_xg_share`).

### Verification & Acceptance Criteria
- [x] `tests/sports/hockey/test_pbp_micro_features.py` & `tests/sports/soccer/test_shot_micro_features.py`:
  - **Score-Adjustment Invariant Test**: Verify that a 5v5 shot taken while leading by `+2` receives a higher score-adjusted weight than an identical shot taken while trailing by `-2`, and that game-total `score_adj_xgf_5v5` remains strictly non-negative.
  - **Open-Play + Set-Piece Conservation Test**: Verify `open_play_xgf + set_piece_xgf + penalty_xgf == total_xgf` (within `1e-6`) for every match.
  - **Strict Temporal Anti-Leakage Test**: Verify mutating shot coordinates, score states, or shift overlaps at event $\ge t$ leaves all pre-game micro-features at $< t$ bit-for-bit unchanged.
- [x] Run `sports-ml backtest --sport hockey` and `sports-ml backtest --sport soccer` against the Milestone 13 closing-line benchmark to verify out-of-sample Log Loss / Brier / RPS improvement.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

# Phase 5: Dual-League MLB & NPB Baseball Plugin (`sports/baseball/`)

Phase 5 extends `sports-analytics-ml` to **Baseball (`--sport baseball --league mlb|npb|all`)**, modeling both **Major League Baseball (MLB, 30 clubs)** and **Nippon Professional Baseball (NPB, 12 Central & Pacific League clubs)** under a unified plugin architecture with league-aware rules:
1. **Stage 1 — League-Aware Calibrated Game Outcome & Run Environment Model**:
   - **MLB (`2-Way Binary`)**: Predicts calibrated $P(\text{Home Win})$ and $P(\text{Away Win}) = 1 - P(\text{Home Win})$ (accounting for unlimited extra innings + Manfred runner) alongside expected runs $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$.
   - **NPB (`3-Way 12-Inning 1X2`)**: Predicts calibrated $P(\text{Home Win})$, $P(\text{Draw / 12-Inning Tie})$, and $P(\text{Away Win})$ summing strictly to $1.0$, plus conditional 2-way Draw-No-Bet (`DNB`) probabilities and expected runs $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$.
2. **Stage 2 — Optional Starting Pitcher & Lineup Batter Matchup Projections (`--include-players`)**:
   - **Starting Pitchers (`SP`)**: Projects `proj_outs_pitched`, `proj_ip`, `proj_k`, `proj_bb`, `proj_hits_allowed`, `proj_er`, `prob_5plus_k`, `prob_6plus_k`, `prob_7plus_k`, and `prob_quality_start`.
   - **Lineup Batters (`1..9`)**: Projects `proj_pa`, `proj_hits`, `proj_total_bases`, `proj_hr`, `proj_runs`, `proj_rbi`, `proj_bb`, `proj_k`, `prob_1plus_hit`, `prob_2plus_tb`, and `prob_1plus_hr`, reconciled with Stage 1 team run totals ($\gamma = 0.50$).

---

## Milestone 16: Core Baseball Extensions, Dual-League Registry (30 MLB + 12 NPB Clubs) & Cross-League Translation Engine (`core/` & `sports/baseball/`)
**Status**: `[x] Completed`

### Goal
Establish the baseball plugin foundation (`src/sports_ml/sports/baseball/`), the 30-club MLB + 12-club NPB registries and ballpark coordinate/factor databases (including NPB regional/provincial neutral stadiums and Koshien displacement rules), exact `outs_pitched` (`scale_factor=27.0`) and `pa` exposure helpers, starter-adjusted & ballpark-adjusted Elo hooks, and the **NPB $\leftrightarrow$ MLB Cross-League Player Translation Engine**.

### Deliverables
- [x] `src/sports_ml/sports/baseball/__init__.py` & `src/sports_ml/sports/baseball/adapter.py`:
  - `BaseballAdapter` implementing `SportAdapter` (`sport_slug = "baseball"`, `topology = "head_to_head"`), validating canonical `games.parquet`, `team_game_logs.parquet`, `pitcher_game_logs.parquet`, and `batter_game_logs.parquet` schemas across `"MLB"` and `"NPB"`, and enforcing sub-day doubleheader ordering (`event_seq` $\in \{1, 2\}$).
- [x] `src/sports_ml/sports/baseball/constants.py`:
  - **MLB 30-Club Registry (`AL` & `NL`)**: Canonical 3-letter codes (`LAD`, `NYY`, `BOS`, `TOR`, `ATL`, `HOU`, `PHI`, `SDP`, `CHC`, `SEA`, `SFG`, `TEX`, `BAL`, `CLE`, `MIN`, `DET`, `KCR`, `TBR`, `CIN`, `MIL`, `STL`, `PIT`, `ARI`, `COL`, `MIA`, `NYM`, `WSN`, `LAA`, `CHW`, `ATH`), historical franchise/code aliases (`OAK`/`ATH`, `TBD`/`TBR`, `FLA`/`MIA`, `MON`/`WSN`, `ANA`/`LAA`, `CLE`), and 30 MLB ballpark coordinates, timezones, roof types (`fixed_dome`, `retractable`, `outdoor`), and multi-year rolling run/HR/LHB/RHB park factors.
  - **NPB 12-Club Registry (`CL` Central League & `PL` Pacific League)**:
    - **Central League (`CL`)**: Yomiuri Giants (`YOM`), Hanshin Tigers (`HAN`), Yokohama DeNA BayStars (`YOK`), Hiroshima Toyo Carp (`HIR`), Tokyo Yakult Swallows (`YAK`), Chunichi Dragons (`CHU`).
    - **Pacific League (`PL`)**: Fukuoka SoftBank Hawks (`SOF`), Orix Buffaloes (`ORX`), Chiba Lotte Marines (`LOT`), Tohoku Rakuten Golden Eagles (`RAK`), Saitama Seibu Lions (`SEI`), Hokkaido Nippon-Ham Fighters (`FUK` / `NHF`).
    - **NPB Primary & Regional Stadium Registry**: Coordinates, dome status, and park factors for all 12 primary stadiums (Tokyo Dome, Koshien, Yokohama, Mazda Zoom-Zoom, Meiji Jingu, Vantelin Dome Nagoya, Mizuho PayPay Dome, Kyocera Dome, Hotto Motto Field Kobe, ZOZO Marine, Rakuten Mobile Park, Belluna Dome, Es Con Field Hokkaido) **plus** NPB provincial/regional neutral venues (Matsuyama Botchan, Miyazaki, Toyama, Nagano, Hard Off Eco Stadium Niigata, Kitakyushu, Akita, Koriyama).
  - **League-Specific Priors & FIP Constants**: Default wOBA linear weights, FIP constant $c_{\text{FIP}}$, HR/FB% priors, and Elo parameters partitioned by `league` $\in \{\text{"MLB"}, \text{"NPB"}\}$ (including NPB 3-way 12-inning draw parameter $\nu_{\text{NPB}} \approx 0.032$).
- [x] `src/sports_ml/sports/baseball/features/translation.py`:
  - **Cross-League Player Translation Priors (`NPB <-> MLB`)**: Transforms prior-season player statistics when a batter or pitcher transitions between `"NPB"` and `"MLB"` (`translate_npb_to_mlb_batter_prior`, `translate_npb_to_mlb_pitcher_prior`, `translate_mlb_to_npb_batter_prior`, `translate_mlb_to_npb_pitcher_prior`) before feeding into `bayesian_shrinkage(...)`, preventing posted NPB stars or foreign NPB signings (`suketto`) from defaulting to generic rookie replacement priors.
- [x] `src/sports_ml/sports/baseball/features/elo_hooks.py`:
  - Implements `StarterModifierHook` ($\Delta E_{\text{SP}} = \kappa_{\text{SP}} \cdot (\mu_{\text{TeamSP, xFIP}} - \text{xFIP}_{\text{SP}})$) and `VenueModifierHook` (ballpark run environment & neutral-venue HFA dampening for NPB regional games and MLB international series).

### Verification & Acceptance Criteria
- [x] `tests/sports/baseball/test_baseball_constants_and_adapter.py`:
  - Verify all 30 MLB and 12 NPB clubs, historical tricode aliases, and primary + provincial stadiums resolve with valid coordinates, timezones, and finite park factors.
  - Verify `BaseballAdapter` validates `event_seq = 1, 2` for same-day doubleheaders and rejects duplicate `(game_id, event_seq)` rows.
  - Verify `outs_pitched` conversion (`6.1 IP -> 19 outs`, `6.2 IP -> 20 outs`) never treats `.1` or `.2` as decimal tenths.
- [x] `tests/sports/baseball/test_cross_league_translation.py`:
  - Verify an NPB ace posting to MLB receives a translated MLB `FIP` / `K%` / `BB%` shrinkage prior strictly stronger than a generic MLB rookie prior, and that current-season MLB batters faced (`BF`) smoothly shrinks the prior weight toward `0` as sample size $n \to \infty$.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 17: Hybrid MLB & NPB Ingestion, Probable Pitchers, Lineups & Silver Lakehouse (`sports/baseball/ingestion/`)
**Status**: `[x] Completed`

### Goal
Build resilient, `.gz`-cached ingestion pipelines for **MLB** (`statsapi.mlb.com` + Statcast/FanGraphs exports) and **NPB** (`npb.jp/bis/eng/`, NPB probable starter announcements `予告先発`, and historical NPB game/player logs), reconciling both leagues into the canonical Silver Parquet tables under `data/silver/baseball/`.

### Deliverables
- [x] `src/sports_ml/sports/baseball/ingestion/mlb_api.py`:
  - Cached `httpx` client (`data/raw/baseball/mlb_api/*.json.gz`) for MLB StatsAPI schedules (`hydrate=probablePitcher,lineups,venue,weather`), boxscores, and active 26-man rosters.
  - Handles split and traditional doubleheaders (`gameNumber` $\rightarrow$ `event_seq` $\in \{1, 2\}$), suspended/resumed games, and postpaid final states (`FINAL` vs. `FUT`).
- [x] `src/sports_ml/sports/baseball/ingestion/statcast.py`:
  - Downloader and Polars parser (`data/raw/baseball/statcast/*.csv.gz`) for MLB team, pitcher, and batter Statcast/FanGraphs game and season-to-date logs (`xwOBA`, `xERA`, `Barrel%`, `HardHit%`, `SwStr%`, `avg_fastball_velo`), with graceful fallback (`has_statcast_metrics = False`) when Statcast lags.
- [x] `src/sports_ml/sports/baseball/ingestion/npb_client.py`:
  - Cached client and parser (`data/raw/baseball/npb/*.json.gz` / `*.csv.gz`) for NPB Central & Pacific League schedules, 12-inning boxscores (explicitly preserving 12-inning Ties `is_tie = 1`), stadium assignments (identifying provincial neutral stadiums and August Koshien road games), and active `ichi-gun` (first-team) roster registrations.
- [x] `src/sports_ml/sports/baseball/ingestion/lineups.py`:
  - **2-Tier Pre-Game Probable Starting Pitcher, 1–9 Batting Order & IL/Roster Resolver**:
    - **Tier 1 (Official Pre-Game Confirmed Lineup & Starter)**: MLB StatsAPI live pre-game batting order (`1..9`) & confirmed `probablePitcher` + NPB pre-game `スタメン` (starting lineup) & official `予告先発` (announced starter).
    - **Tier 2 (Daytime / Early Projection Fallback)**: Probable starter rotation resolver + projected 1–9 platoon batting order (using handedness-specific lineup templates vs. `RHP` vs. `LHP`, excluding MLB `10/15/60-day IL` and NPB deactivated farm players).
- [x] `src/sports_ml/sports/baseball/ingestion/health.py` & `reconciler.py`:
  - `check_baseball_data_sources(league: Literal["mlb", "npb", "all"] = "all")`: Upstream reachability and schema contract probes for `sports-ml check-sources --sport baseball`.
  - `BaseballSilverReconciler`: Normalizes and upserts MLB and NPB records into `zstd`-compressed `data/silver/baseball/games.parquet`, `team_game_logs.parquet`, `pitcher_game_logs.parquet`, and `batter_game_logs.parquet`.

### Verification & Acceptance Criteria
- [x] `tests/sports/baseball/test_baseball_ingestion.py`:
  - Test MLB StatsAPI schedule/boxscore parsing (including a same-day doubleheader with `event_seq = 1` and `event_seq = 2`), Statcast join + fallback, and NPB 12-inning tie game parsing (`home_runs == away_runs`, `innings_played == 12`, `outcome_1x2 == 1`).
  - Test 2-Tier pre-game resolver for both confirmed 1–9 lineups and daytime platoon-projected lineups (verifying different projected lineups when facing a `LHP` vs. `RHP`).
  - Verify all raw caches are written as `.json.gz` / `.csv.gz` via `core/cache.py` and Silver tables enforce uniqueness on `(game_id, event_seq)`.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 18: Baseball Pre-Game Feature Engineering: Starting Pitchers, Bullpen Workload, Platoon Lineups & Venue Context (`sports/baseball/features/`)
**Status**: `[x] Completed`

### Goal
Construct the leak-free Gold feature pipelines (`data/gold/baseball/`) capturing starting pitcher skill & pitch-count ceilings, 3-day bullpen fatigue & high-leverage availability, handedness platoon-adjusted 1–9 batting orders, cross-league translation priors, and ballpark/schedule context for both MLB and NPB.

### Deliverables
- [x] `src/sports_ml/sports/baseball/features/pitcher.py`:
  - Pre-game starting pitcher feature builder using `.shift(1)` over `(start_time_utc, event_seq, game_id)`:
  - Rolling 5-start, 10-start, EWMA, and Bayesian-shrunk ($k_{\text{sp}} = 180\text{ BF}$, with `NPB <-> MLB` translation support) `sp_xfip_shrunk`, `sp_fip_shrunk`, `sp_era_shrunk`, `sp_k_minus_bb_pct_shrunk`, `sp_whip_shrunk`, `sp_hr_per_9_shrunk`, `sp_gb_pct_shrunk`, `sp_days_rest`, `sp_rolling_pitches_per_start`, and `sp_expected_outs_ceiling`.
- [x] `src/sports_ml/sports/baseball/features/bullpen.py`:
  - Pre-game relief corps quality (`bp_xfip_shrunk`, `bp_k_minus_bb_pct_shrunk`, `bp_lob_pct_shrunk`) and acute workload fatigue strictly before first pitch (`bp_pitches_last_1d`, `bp_pitches_last_3d`, `bp_outs_last_3d`, and `high_leverage_arms_unavailable_count` for top-3 relievers who pitched back-to-back days or $\ge 28$ pitches yesterday).
- [x] `src/sports_ml/sports/baseball/features/lineup.py`:
  - Active 1–9 batting order aggregator weighting batters by slot Plate Appearance weights ($w_1 > w_2 > \dots > w_9$) and adjusting each batter's shrunk `wOBA`, `ISO`, `K%`, and `BB%` for the opposing starting pitcher's handedness (`opp_sp_hand` $\in \{\text{"L"}, \text{"R"}\}$ with heavy platoon shrinkage $k_{\text{LHB}} = 1000, k_{\text{RHB}} = 2200$).
  - Computes `lineup_platoon_woba`, `lineup_platoon_iso`, `lineup_platoon_k_pct`, `lineup_same_hand_share`, and `missing_regular_pa_woba`.
- [x] `src/sports_ml/sports/baseball/features/team.py` & `player.py`:
  - Team rolling run-scoring efficiency, Pythagorean / BaseRuns luck regression signal (`actual_win_pct - pythag_win_pct_w30`, `runs_for - baseruns_for_w20`), ballpark run/HR factors (`venue_run_factor`, `venue_hr_factor_matchup`), dome/weather context, travel/timezone shift, doubleheader flags (`is_doubleheader_g1`, `is_doubleheader_g2`), interleague flag (`is_interleague`), and NPB neutral/Koshien road-trip flags.
  - `player.py`: Stage 2 individual starting pitcher and lineup batter matchup feature matrix builder.

### Verification & Acceptance Criteria
- [x] `tests/sports/baseball/test_baseball_features.py`:
  - **Strict Anti-Leakage Test (Including Same-Day Doubleheader Game 1 $\rightarrow$ Game 2)**: Verify that mutating runs, hits, pitcher strikeouts, or bullpen pitches for events at index $\ge t$ (including Game 2 of a same-day doubleheader at index $t$) produces **bit-for-bit identical** features for Game 1 at index $t-1$.
  - **Platoon Handedness Sensitivity Test**: Verify that switching an opposing starting pitcher from a right-hander (`RHP`) to a left-hander (`LHP`) updates `lineup_platoon_woba` and `lineup_same_hand_share` deterministically without leaking current-game stats.
  - **Bullpen 3-Day Fatigue Test**: Verify that a reliever pitching on Day $d-2$ and Day $d-1$ increments `high_leverage_arms_unavailable_count` on Day $d$, and resets cleanly after an off-day on Day $d+1$.
  - **Cold-Start & Cross-League Transfer Test**: Verify Opening Day (Game 1), debuting rookies, and NPB $\rightarrow$ MLB transferred players produce finite, non-null features with zero `NaN`s.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 19: Stage 1 — League-Aware Calibrated Game Outcome (2-Way MLB & 3-Way 12-Inning NPB) & Run Environment Pipeline (`sports/baseball/models/stage1_game.py`)
**Status**: `[x] Completed`

### Goal
Implement the mandatory Stage 1 baseball game outcome pipeline supporting **2-Way Binary Win Probability for MLB** and **3-Way (`Home Win`, `12-Inning Tie/Draw`, `Away Win`) Probability for NPB**, coupled with Negative-Binomial/Tweedie team-run environment estimators $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ and walk-forward backtesting.

### Deliverables
- [x] `src/sports_ml/sports/baseball/models/stage1_game.py`:
  - **Auxiliary Team-Runs Environment Model**: Negative-Binomial / Tweedie regressors (`objective="tweedie"`, `tweedie_variance_power=1.5`) estimating pre-game expected runs $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ and dispersion $\alpha_{\text{league}}$, capturing baseball's overdispersed (multi-run home run / big-inning) scoring distribution.
  - **Bivariate Negative-Binomial / Skellam Run-Distribution Probabilities**:
    - Computes 9-inning regulation win/tie/loss probabilities and models extra-inning resolution:
      - **MLB**: Unlimited extra innings (with Manfred runner scoring boost in innings $10+$) $\rightarrow$ strictly binary $(P_{\text{NB}}(\text{Home}), P_{\text{NB}}(\text{Away}))$ summing to $1.0$.
      - **NPB**: Explicit 3-inning extra cap (innings $10, 11, 12$) $\rightarrow$ structural 3-way $(P_{\text{NB}}(\text{Home}), P_{\text{NB}}(\text{Draw}_{12}), P_{\text{NB}}(\text{Away}))$ summing to $1.0$.
  - **League-Aware Hybrid GBDT + Calibration Ensemble (`Stage1BaseballGameModel`)**:
    - Blends Starter-Adjusted Elo, Negative-Binomial run-implied probabilities, and LightGBM classifiers trained on pre-game starting pitcher, bullpen, platoon lineup, and ballpark differentials.
    - Applies out-of-fold `BinaryProbabilityCalibrator` for MLB (`P(Home) + P(Away) = 1.0`) and `MulticlassProbabilityCalibrator` for NPB (`P(Home) + P(Draw) + P(Away) = 1.0`, plus derived Draw-No-Bet `p_home_dnb = P(Home) / (P(Home) + P(Away))`).
  - **Walk-Forward Backtesting Harness (`run_baseball_stage1_walk_forward_backtest`)**:
    - Evaluates MLB via **Log Loss**, **Brier Score**, **ECE**, and Accuracy against coin-flip (`0.50`), empirical MLB home-field baseline (`~0.535`), and Starter-Adjusted Elo.
    - Evaluates NPB via **Ranked Probability Score (RPS)**, **3-Class Log Loss**, **Multi-Class Brier Score**, **ECE**, and **DNB Log Loss** against empirical NPB `1X2` prior and NPB Elo-1X2 baseline, with `FoldBoundaryAudit` proving zero temporal overlap.

### Verification & Acceptance Criteria
- [x] `tests/sports/baseball/test_stage1_model.py`:
  - Verify MLB predictions satisfy $P(\text{Home Win}) \in (0, 1)$, $P(\text{Away Win}) \in (0, 1)$, `prob_draw == 0.0`, and $P(\text{Home Win}) + P(\text{Away Win}) = 1.0$ (within `1e-9`).
  - Verify NPB predictions satisfy $P(\text{Home Win}), P(\text{Draw}), P(\text{Away Win}) \in (0, 1)$ and $P(\text{Home Win}) + P(\text{Draw}) + P(\text{Away Win}) = 1.0$ (within `1e-9`), and that $P(\text{Draw})$ increases when $\hat{\lambda}_{\text{home}} \approx \hat{\lambda}_{\text{away}}$ in low-run pitching duels (e.g., $\hat{\lambda} = 2.8$ vs. $\hat{\lambda} = 5.5$).
  - Verify walk-forward training never fits the GBDT, run regressors, or calibrators on validation/test dates (`gbdt_train_max_time < calibrator_min_time <= calibrator_max_time < val_min_time`).
  - Verify calibrated models beat the empirical home-win/1X2 baseline on synthetic directional MLB and NPB datasets.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 20: Stage 2 Starting Pitcher & Batter Projections, Unified Baseball Predictor & CLI Integration (`sports/baseball/models/` & `cli.py`)
**Status**: `[x] Completed`

### Goal
Implement the optional Stage 2 Starting Pitcher (`SP`) and Lineup Batter (`1..9`) matchup projection models conditioned on Stage 1, wire `BaseballMatchupPredictor` into the `sports-ml` CLI (`--sport baseball [--league mlb|npb|all]`), and update platform documentation and skills.

### Deliverables
- [x] `src/sports_ml/sports/baseball/models/stage2_player.py`:
  - **Starting Pitcher (`SP`) Projection Pipeline**:
    - Predicts expected outs recorded `proj_outs_pitched` ($\in [0, 27]$), `proj_ip` ($\text{outs}/3$), and `proj_batters_faced` ($\widehat{\text{BF}}$).
    - LightGBM Poisson/Tweedie regressors conditioned on starter `K%`/`BB%`/`xFIP`, opposing lineup platoon `K%`/`wOBA`, ballpark factor, and Stage 1 $\hat{\lambda}_{\text{opp}}$ predicting `proj_k`, `proj_bb`, `proj_hits_allowed`, `proj_er`, `prob_5plus_k`, `prob_6plus_k`, `prob_7plus_k`, and `prob_quality_start`.
  - **Lineup Batter (`1..9`) Projection Pipeline**:
    - **Slot & Home/Away PA Projector**: Estimates `proj_pa` ($\in [0, 6.0]$) conditioned on `batting_order_slot` (`1..9`), `is_home` (accounting for unplayed bottom-9th innings when home team leads), and Stage 1 $\hat{\lambda}_{\text{team}}$.
    - **Batter Count & Milestone Models**: Predicts `proj_hits`, `proj_total_bases`, `proj_hr`, `proj_runs`, `proj_rbi`, `proj_bb`, `proj_k`, `prob_1plus_hit`, `prob_2plus_tb`, and `prob_1plus_hr` conditioned on opposing `SP` skill/handedness, opposing bullpen `xFIP`, and ballpark HR/run factors.
  - **Cross-Stage Coherence Reconciler (`reconcile_baseball_stage2_projections`)**:
    - Soft-scales lineup batter `proj_runs` and `proj_rbi` toward Stage 1's $\hat{\lambda}_{\text{team}}$ ($\gamma = 0.50$) and reconciles starter `proj_er` with Stage 1 opponent scoring pace.
- [x] `src/sports_ml/sports/baseball/models/predictor.py`:
  - `BaseballMatchupPredictor` supporting `predict_matchup(..., league: str | None = None, event_seq: int = 1, include_players: bool = False)`, `predict_date(..., league: str = "all", include_players: bool = False)`, and `predict_team_next_games(team, n_games, include_players=False)` (auto-resolving whether `team` is an MLB or NPB club).
  - Surfaces pre-game confidence warnings (`CONFIRMED_SP` vs. `PROBABLE_SP` vs. `TBD_ROTATION_PROJECTED`, `CONFIRMED_LINEUP_1_9` vs. `PLATOON_PROJECTED_LINEUP`, `CROSS_LEAGUE_TRANSLATION_PRIOR`, `NPB_NEUTRAL_PROVINCIAL_STADIUM`, `STATCAST_FALLBACK`), and appends to `data/predictions/baseball_predictions.parquet`.
- [x] `src/sports_ml/cli.py`:
  - Register `--sport baseball` and add `--league [mlb|npb|all]` option across `check-sources`, `ingest`, `features build`, `train`, `backtest`, `predict` (including `--matchup AWAY@HOME`, `--game-seq 1|2` for doubleheaders, `--team`, `--games`, `--include-players`, `--offline`), and `score-predictions`.
  - Rich terminal tables displaying MLB 2-way (`P(Away)`, `P(Home)`) and NPB 3-way (`P(Away)`, `P(Draw)`, `P(Home)`, `P(Home DNB)`), Starting Pitcher matchup cards (`SP Hand`, `Shrunk xFIP`, `K-BB%`, `Proj IP`, `Proj K`, `P(QS)`), and 1–9 Batting Order projection tables.

### Verification & Acceptance Criteria
- [x] `tests/sports/baseball/test_stage2_player.py`:
  - Verify all predicted pitcher/batter counts, `proj_outs_pitched` ($\in [0, 27]$), and `proj_pa` ($\in [0, 6.5]$) are non-negative ($\ge 0$) and milestone probabilities (`prob_6plus_k`, `prob_quality_start`, `prob_1plus_hit`, `prob_1plus_hr`) lie in $[0, 1]$.
  - Verify **Batting Order Slot Monotonicity**: For two identical batters on the same team, Slot 1 (`leadoff`) receives strictly higher `proj_pa` than Slot 9.
  - Verify **Stage-Conditioning & Park-Factor Monotonicity**: Increasing Stage 1's $\hat{\lambda}_{\text{team}}$ or moving a game from a pitcher's dome (`venue_run_factor = 0.86`) to Coors Field / Jingu (`venue_run_factor = 1.25`) increases reconciled batter run/TB projections; facing a high-`K%` ace increases opposing batter `proj_k` and starter `prob_6plus_k`.
- [x] `tests/sports/baseball/test_baseball_cli_e2e.py`:
  - End-to-end offline CLI integration test running `ingest` $\rightarrow$ `features build` $\rightarrow$ `train` $\rightarrow$ `backtest` $\rightarrow$ `predict` (Stage 1 & `--include-players`, across both `--league mlb` and `--league npb`, plus `--team` next-$N$-games) $\rightarrow$ `score-predictions`.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 21: MLB Statcast Pitch/Batted-Ball Micro-Features, NPB DIPS/Koshien Context & Closing Odds Firewall (`sports/baseball/`)
**Status**: `[x] Completed`

### Goal
Enrich the baseball plugin with pitch- and batted-ball micro-features (**MLB Statcast `xwOBA`, `Barrel%`, `HardHit%`, `Stuff+` / Arsenal Velocity Delta & Whiff Rate**; **NPB DIPS, Groundball/Flyball & Provincial/Koshien Venue Micro-Features**) and benchmark Stage 1 MLB & NPB predictions against historical closing odds under the strict **`market_*` anti-contamination firewall**.

### Deliverables
- [x] `src/sports_ml/sports/baseball/ingestion/statcast_micro.py` & `npb_micro.py`:
  - Aggregate pitch-level and batted-ball micro-summaries into compact `zstd`-compressed `data/silver/baseball/pitcher_micro_logs.parquet` and `batter_micro_logs.parquet`:
    - **Pitcher Arsenal & Stuff Micro-Features**: Fastball velocity delta vs. 12-month baseline (`sp_velo_delta_mph`, detecting acute fatigue or injury Decline before ERA spikes!), swinging-strike rate (`sp_swstr_pct`), chase rate (`sp_o_swing_pct`), contact suppression (`sp_xwoba_against`, `sp_barrel_pct_against`), and First-Pitch Strike % (`sp_f_strike_pct`).
    - **Batter Contact Quality & Discipline Micro-Features**: `batter_xwoba`, `batter_barrel_per_pa`, `batter_hard_hit_pct`, `batter_sweet_spot_pct`, `batter_chase_pct`, and `batter_woba_minus_xwoba_luck` (identifying unsustainable BABIP streaks prior to regression).
    - **NPB DIPS, Batted-Ball & Koshien/Provincial Micro-Context**: Infield/outfield hit & HR park-interaction scores, sacrificial bunt / small-ball tactical propensity (`sac_bunt_rate`), and neutral provincial stadium travel/surface adjustments.
- [x] `src/sports_ml/sports/baseball/ingestion/odds.py`:
  - `BaseballOddsIngestor`: Imports historical and pre-game evaluation-only closing odds for MLB (2-way moneyline & run total) and NPB (3-way `1X2` and 2-way Draw-No-Bet moneyline) into `games.parquet` (`market_home_decimal_odds`, `market_draw_decimal_odds`, `market_away_decimal_odds`, `market_prob_home`, `market_prob_draw`, `market_prob_away`) using power de-vigging from `core/market.py`.
- [x] `src/sports_ml/sports/baseball/features/pipeline.py`:
  - Wire rolling `.shift(1)` micro-features into Stage 1 and Stage 2 (`diff_sp_velo_delta`, `diff_sp_swstr_pct`, `diff_lineup_xwoba`, `diff_woba_minus_xwoba_regression`).
  - Enforce the **Hard Odds Firewall** prohibiting any `market_*` or `odds` column from entering `STAGE1_FEATURE_COLUMNS` or `STAGE2_FEATURE_COLUMNS`.

### Verification & Acceptance Criteria
- [x] `tests/sports/baseball/test_baseball_micro_features.py`:
  - **Velocity Delta & BABIP Regression Test**: Verify that a `-1.8 mph` drop in a starter's fastball velocity over their last 2 starts is captured in pre-game `sp_velo_delta_mph` at start $t$, and that mutating pitch tracking at start $\ge t$ leaves all features at $< t$ bit-for-bit unchanged.
- [x] `tests/sports/baseball/test_baseball_odds_firewall.py`:
  - **Strict Anti-Contamination Test**: Verify that no column starting with `market_` or containing `odds` is ever present in Baseball Stage 1 or Stage 2 feature lists, and that mutating `market_*` columns in `games.parquet` changes `0.0` of predicted probabilities while updating the backtest `MarketEdgeReport`.
- [x] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

# Phase 6: Football Plugin (`sports/football/`, NFL First)

Phase 6 extends `sports-analytics-ml` to **Football (`--sport football --league nfl`)** under the same two-stage architecture, sourced primarily from the free **nflverse** data releases (schedules, play-by-play with EPA/WP, weekly player stats, rosters, injuries, depth charts, snap counts, Next Gen Stats, draft picks). The plugin lives at `src/sports_ml/sports/football/` (slug `football`, mirroring baseball's `--league` pattern): `--league nfl` is the only league implemented and the default; the `league` column and per-league constants/registries keep the door open for `ncaaf` later without restructuring.

**NFL-specific design constraints** (every milestone below must honor them):
1. **Tiny samples**: 17 games/team/season (16 before 2021) and ~272 games/season. All team/player rates use heavy Bayesian shrinkage with cross-season carry-over, and calibrators are fit on **pooled prior seasons** using parametric (Platt/temperature) calibration, not isotonic.
2. **The starting QB dominates outcomes**: QB identity is a first-class pre-game input (QB-adjusted Elo, QB-level priors that follow the player across teams, team-level priors that follow the roster/scheme).
3. **Weekly cadence**: one game per team per week (`event_seq = 1` always), bye weeks, Thursday/Monday/international/neutral-site games, and a playoff bracket (`WC`, `DIV`, `CON`, `SB`) ordered strictly by `(start_time_utc, event_seq, game_id)`.
4. **Ties** (~0.2% of games) are a **2-way binary** outcome: ties count as `0.5` in Elo/rating updates and are **excluded from classifier targets and calibration** (never a third class).
5. **Stage 1 outputs**: calibrated $P(\text{Home Win})$ and $P(\text{Away Win}) = 1 - P(\text{Home Win})$ plus expected team scores $(\hat{\mu}_{\text{home}}, \hat{\mu}_{\text{away}})$.
6. **Stage 2 (optional, `--include-players`)**: QB, RB, and WR/TE stat projections plus milestone and **anytime-TD** probabilities.
7. **Post-game fields in nflverse are leakage traps**: schedule columns such as `spread_line`, `total_line`, `home_moneyline`, `temp`, `wind`, `home_score`, and post-game inactives/injury designations must be routed to `market_*` (evaluation-only), forecast-vs-observed handling (Milestone 27), or the label side—never silently into feature tables.

---

## Milestone 22: Football Plugin Foundation, NFL 32-Team Registry, Stadium Database & QB-Adjusted Elo (`core/` & `sports/football/`)
**Status**: `[ ] Not Started`

### Goal
Establish the football plugin skeleton (league-aware, NFL first), canonical Silver schemas, 32-team/stadium registry, weekly-cadence schedule primitives, and a margin-of-victory, QB-adjusted Elo hook—without rewriting existing `core/` abstractions (only additive extensions).

### Deliverables
- [ ] `docs/football-data-and-features.md`: Data-source map (nflverse release assets consumed directly as parquet/csv.gz, optionally via `nflreadpy`; do **not** depend on the deprecated `nfl_data_py`), Silver/Gold schemas, exact feature math, per-source availability windows (play-by-play 1999+, injuries ~2009+, snap counts 2012+, Next Gen Stats 2016+), temporal boundary rules, and the pre-game availability tiers defined in Milestone 23. Also update `docs/architecture.md` (§4 abstraction table, directory tree, CLI section) and `README.md` (supported sports, CLI examples) to include NFL.
- [ ] `src/sports_ml/sports/football/__init__.py` & `adapter.py`:
  - `FootballAdapter` implementing `SportAdapter` (`sport_slug = "football"`, `topology = "head_to_head"`), validating a `league` column (`"NFL"`; `"NCAAF"` reserved/unimplemented) and canonical `games.parquet`, `team_game_logs.parquet`, and `player_game_logs.parquet` schemas, enforcing `event_seq == 1`, one game per team per `(season, week)`, and valid `season_type` $\in \{\text{"REG"}, \text{"WC"}, \text{"DIV"}, \text{"CON"}, \text{"SB"}\}$ (and `"PRE"` rejected from model tables).
- [ ] `src/sports_ml/sports/football/constants.py` (NFL registries and parameters partitioned by `league`):
  - **32-Team Registry (`AFC`/`NFC` × `East`/`North`/`South`/`West`)** with canonical nflverse tricodes and historical aliases (`OAK`→`LV`, `SD`→`LAC`, `STL`→`LA`/`LAR`, `WSH`/`WAS`, `JAC`/`JAX`, `ARZ`/`ARI`, `BLT`/`BAL`, `CLV`/`CLE`, `HST`/`HOU`, `SL`/`LA`), with franchise-continuity mapping so relocated clubs keep their rating history.
  - **Stadium Registry**: coordinates, timezone, `roof` $\in \{\text{"dome"}, \text{"closed\_retractable"}, \text{"open\_retractable"}, \text{"outdoors"}\}$, `surface` $\in \{\text{"grass"}, \text{"turf"}\}$, altitude (Denver), plus **neutral-site/international venues** (London, Munich, Frankfurt, Mexico City, São Paulo, Madrid, Super Bowl hosts) with `is_neutral_site` and `is_international` flags.
  - **Elo & Prior Parameters**: initial rating 1500, $K \approx 20$, home-field advantage $\approx 48$ Elo (neutral-site $= 0$), season-to-season regression toward the mean ($\approx 1/3$), playoff $K$ multiplier, and league-average points/EPA priors per season (passing-era drift handled by per-season league means).
- [ ] `src/sports_ml/core/schedule.py` (additive) & `src/sports_ml/sports/football/features/schedule_hooks.py`:
  - Weekly-cadence helpers: `days_rest` (exact kickoff-to-kickoff hours), `is_short_week` ($\le 5$ days), `is_post_bye` ($\ge 13$ days), `is_post_mini_bye`, `prior_game_was_overtime`, plus cross-timezone travel/body-clock kickoff (`kickoff_local_hour_vs_body_clock`) and `is_international_travel`. Core stays sport-agnostic; NFL-specific rules (e.g., Thursday Night Football) live in the hook.
- [ ] `src/sports_ml/sports/football/features/elo_hooks.py`:
  - **Margin-of-Victory Elo** (FiveThirtyEight-style multiplier $\ln(|\Delta_{\text{pts}}| + 1) \cdot \frac{2.2}{0.001\,|\Delta_{\text{Elo}}| + 2.2}$ to damp autocorrelation) with ties updating at `0.5`.
  - `StarterModifierHook` implementation for the **starting QB**: $\Delta E_{\text{QB}} = \kappa_{\text{QB}} \cdot (\text{QBValue}_{\text{starter}} - \text{QBValue}_{\text{team baseline}})$ where `QBValue` is a shrunk rolling EPA+CPOE composite supplied by Milestone 24 (hook accepts any callable returning a finite value, so M22 is testable with a stub).
  - `VenueModifierHook`: HFA scaling for neutral/international sites, dome/altitude, and rest differential adjustments.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_football_constants_and_adapter.py`:
  - Verify all 32 teams, 8 divisions, historical aliases, and every stadium (incl. international/neutral sites) resolve with valid coordinates, timezones, roof/surface enums, and finite altitude.
  - Verify `FootballAdapter` rejects an unknown `league`, `event_seq != 1`, duplicate `(game_id)`, two games for one team in one `(season, week)`, and `PRE` season rows.
- [ ] `tests/sports/football/test_football_schedule_and_elo.py`:
  - Verify short-week / post-bye / mini-bye flags on a fixture schedule (Thursday after Sunday $=$ 4 days; bye $\ge 13$ days).
  - Verify Elo conservation (zero-sum updates), tie $=$ `0.5` update, neutral-site HFA $= 0$, offseason regression toward 1500, and that a QB downgrade (starter → backup) lowers the team's pre-game rating by exactly $\kappa_{\text{QB}} \cdot \Delta\text{QBValue}$.
  - **Anti-Leakage**: mutating the score of any game at index $\ge t$ leaves pre-game ratings for games $< t$ bit-for-bit unchanged.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 23: nflverse Ingestion, Starting-QB / Inactives Resolver & Silver Lakehouse (`sports/football/ingestion/`)
**Status**: `[ ] Not Started`

### Goal
Build resilient `.gz`-cached ingestion for the nflverse releases and reconcile them into canonical Silver Parquet tables under `data/silver/football/`, with an explicit **as-of timestamp policy** so every pre-game signal is only what was knowable before kickoff.

### Deliverables
- [ ] `src/sports_ml/sports/football/ingestion/nflverse_client.py`:
  - Cached `httpx` client (`data/raw/football/nflverse/*.parquet|*.csv.gz` via `core/cache.py`) for nflverse release assets: `schedules/games`, `pbp_{YYYY}`, `player_stats_{YYYY}` (weekly), `rosters` & `rosters_weekly`, `injuries_{YYYY}`, `depth_charts_{YYYY}`, `snap_counts_{YYYY}`, `nextgen_stats` (passing/rushing/receiving), `draft_picks`, and `players` (ID crosswalk: `gsis_id` canonical, with `pfr_id`/`espn_id` mappings).
  - Supports `--from-year`, `--current-season`, and date-window sync; handles in-season weekly refreshes (late stat corrections) via idempotent upserts; degrades gracefully when a source lags (`has_snap_counts`, `has_ngs`, `has_injury_report` flags).
- [ ] `src/sports_ml/sports/football/ingestion/lineups.py`:
  - **2-Tier Pre-Game Availability & Starting-QB Resolver**:
    - **Tier 1 (Official, $\approx 90$ min pre-kickoff)**: Official inactives list + confirmed starting QB (first offensive snap QB from play-by-play for historical games; the as-of snapshot in live mode).
    - **Tier 2 (Early-Week Projection)**: Latest depth chart + Wed–Fri practice report designations (`DNP`/`LP`/`FP`) and game status (`Out`/`Doubtful`/`Questionable`) → probabilistic starting-QB and key-player availability (`p_starts`), **never** using post-game inactives for a Tier 2 prediction.
  - Every snapshot row carries `as_of_time_utc`; historical backfill must reconstruct features using only snapshots with `as_of_time_utc < kickoff_utc` (kickoff $-$ 90 min for Tier 1).
- [ ] `src/sports_ml/sports/football/ingestion/health.py` & `reconciler.py`:
  - `check_football_data_sources(league: Literal["nfl"] = "nfl")` for `sports-ml check-sources --sport football --league nfl` (reachability + schema-contract probes on each release asset).
  - `FootballSilverReconciler`: Normalizes franchise aliases, builds `games.parquet` (one row per game; labels `home_score`, `away_score`, `is_tie`, `overtime`), `team_game_logs.parquet` (aggregated from play-by-play: plays, dropbacks, EPA, success, yards, turnovers, penalties, time of possession, points), `player_game_logs.parquet` (weekly player stats + snap share, targets, carries), `team_week_availability.parquet` (as-of injury/inactive snapshots), and `qb_game_logs.parquet`, all `zstd` Parquet.
  - **Leakage Quarantine**: schedule-embedded `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`, `temp`, `wind`, and `roof` observation columns are split out at reconcile time (odds → `market_*` namespace; observed weather → `obs_weather_*`), so no downstream feature builder can read them by accident.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_football_ingestion.py` (offline fixtures, trimmed nflverse samples):
  - Test schedule/PBP/player-stat parsing, overtime and tie games (`home_score == away_score`, `is_tie == 1`), neutral-site and international games, playoff round mapping, relocated-franchise alias normalization, and `gsis_id` crosswalk joins.
  - Verify raw caches are written compressed through `core/cache.py`, Silver tables enforce uniqueness on `game_id` / `(game_id, team)` / `(game_id, gsis_id)`, and re-ingesting a corrected week is idempotent.
  - **As-Of Test**: a Tier 2 resolver call at `as_of_time_utc = Friday 17:00` never reads the Sunday inactives list; the Tier 1 resolver never reads data stamped after kickoff $-$ 90 min.
  - **Quarantine Test**: assert Silver `games.parquet` exposes odds only as `market_*` and observed weather only as `obs_weather_*`.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 24: NFL Pre-Game Feature Engineering: Team EPA, Starting QB, Rest/Schedule & Cross-Season Shrinkage (`sports/football/features/`)
**Status**: `[ ] Not Started`

### Goal
Construct leak-free Gold features (`data/gold/football/`) capturing opponent-adjusted team efficiency, starting-QB quality, schedule/rest context, and a principled **cold-start policy** for a 17-game season.

### Deliverables
- [ ] `src/sports_ml/sports/football/features/team.py`:
  - Strict `.shift(1)` rolling over `(start_time_utc, event_seq, game_id)` (rolling 4/8-game windows + EWMA + season-to-date), exposure-normalized via `pregame_exposure_rate` (per play / per dropback):
    - Offense & defense `epa_per_play`, `pass_epa_per_dropback`, `rush_epa_per_rush`, `success_rate`, `early_down_success_rate`, `explosive_play_rate`, `sack_rate`, `turnover_rate`, `penalty_yards_per_game`, `points_per_drive` (garbage-time filtered with a win-probability band, e.g., $0.05 \le WP \le 0.95$ and exclusion of kneel/spike plays).
  - **Opponent adjustment**: ridge-regularized strength-of-schedule adjustment of EPA ratings computed only from games $< t$.
  - **Luck/regression signals**: turnover differential vs. expected, fumble-recovery luck, close-game record (one-score games), Pythagorean expectation (`exponent ≈ 2.37`), and red-zone TD% regression.
  - **Cold-start / season-boundary policy**: Weeks 1–4 blend prior-season (and multi-year) team ratings with current-season rolling values using Bayesian shrinkage in *games played* ($k_{\text{team}} \approx 8$ games), with offseason regression toward the mean, roster-continuity weighting (returning snap share, QB change, head-coach change), and **no hard-coded week cutoff**.
- [ ] `src/sports_ml/sports/football/features/qb.py`:
  - Starting-QB features from the Tier 1/Tier 2 resolver: shrunk rolling `qb_epa_per_dropback`, `qb_cpoe`, `qb_sack_rate`, `qb_air_yards_per_attempt`, `qb_rush_epa`, `qb_games_with_team`, `qb_is_rookie_or_debut`, `qb_is_backup`, and `qb_value` (composite feeding the M22 Elo hook).
  - **Player-level prior follows the QB across teams**; debut/rookie QBs use a draft-capital prior (`draft_round`, `draft_pick` from nflverse `draft_picks`) shrunk toward the league replacement-level QB.
- [ ] `src/sports_ml/sports/football/features/schedule_context.py`:
  - `days_rest`, `rest_diff`, `is_short_week`, `is_post_bye`, `is_thursday_game`, `is_monday_game`, `is_international`, `is_neutral_site`, `is_division_game`, `is_conference_game`, `is_playoff`, `playoff_round`, `travel_km`, `timezone_shift`, `consecutive_road_games`, plus Elo ratings & QB-adjusted Elo differentials.
- [ ] `src/sports_ml/sports/football/features/pipeline.py`:
  - Builds `data/gold/football/stage1_features.parquet` with home-minus-away `diff_*` columns, defines `STAGE1_FEATURE_COLUMNS`/`STAGE2_FEATURE_COLUMNS`, and enforces the **Hard Odds & Observed-Outcome Firewall** (rejecting any column starting with `market_` / `obs_weather_` / containing `odds`, `spread`, or `closing`).
  - Missingness is explicit (`*_is_imputed` flags) with **zero `NaN`s** for Week 1, debut QBs, and relocated franchises.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_football_features.py`:
  - **Strict Anti-Leakage Test**: Mutating plays, scores, EPA, or player stats for games at index $\ge t$ yields **bit-for-bit identical** features for games $< t$ (covering Thursday $\rightarrow$ Sunday $\rightarrow$ Monday ordering within a week).
  - **Bye-Week Test**: A team on bye has no row that week; next-game `days_rest` $\ge 13$ and rolling windows skip the bye without introducing gaps or double-counting.
  - **Cold-Start Test**: Week 1, expansion-style new-QB, and relocated-franchise cases produce finite, non-null features; the weight on current-season data increases monotonically with games played and tends to `1` as $n \to \infty$.
  - **QB-Swap Test**: Replacing the starter with a backup deterministically changes `qb_*` and `qb_value` features while leaving all team rolling features unchanged.
  - **Garbage-Time Filter Test**: Plays outside the WP band are excluded from `epa_per_play` and kneels/spikes never enter success-rate denominators.
- [ ] `tests/sports/football/test_football_odds_firewall.py`: no `market_*`, `obs_weather_*`, or odds-like column ever appears in either feature list, and mutating those Silver columns changes `0.0` of downstream features.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 25: Stage 1 — Calibrated Home/Away Win Probability & Expected Team Scores (`sports/football/models/stage1_game.py`)
**Status**: `[ ] Not Started`

### Goal
Implement the mandatory NFL Stage 1 pipeline: calibrated 2-way win probabilities and expected team scores, with a walk-forward backtest that respects NFL's small samples and season-week structure.

### Deliverables
- [ ] `src/sports_ml/sports/football/models/stage1_game.py`:
  - **Expected Team Scores Model**: Poisson/Tweedie-style or Gaussian-link LightGBM regressors (heavily regularized, shallow trees, monotone constraints on key QB/EPA features) predicting $(\hat{\mu}_{\text{home}}, \hat{\mu}_{\text{away}})$ with league/season mean anchoring; reports implied margin $\hat{\mu}_{\text{home}} - \hat{\mu}_{\text{away}}$ and total $\hat{\mu}_{\text{home}} + \hat{\mu}_{\text{away}}$ for diagnostics.
  - **Hybrid Win-Probability Ensemble (`Stage1FootballGameModel`)**: Blends QB-adjusted Elo, a score-implied win probability $\Phi(\text{margin}/\sigma_{\text{margin}})$ with empirically estimated $\sigma_{\text{margin}} \approx 13$–$14$, and a regularized logistic / shallow-GBDT classifier. Ties are dropped from classifier targets and calibration; ratings treat them as `0.5`.
  - **Calibration**: `BinaryProbabilityCalibrator` using **Platt / temperature scaling** fit on pooled out-of-fold predictions from prior seasons (no isotonic—too few games), with a guard that falls back to the identity map when calibration sample size is below a configured minimum.
  - **Walk-Forward Backtest Harness (`run_football_stage1_walk_forward_backtest`)**: Expanding-window, week-aligned splits (train $\rightarrow$ calibrate $\rightarrow$ test on whole future seasons or season-week blocks) with `FoldBoundaryAudit`; evaluates **Log Loss**, **Brier Score**, **ECE**, accuracy, and **score MAE / RMSE** against coin-flip, empirical home-win-rate prior (reported per era), and QB-adjusted Elo baselines; includes playoff-only and Week 1–4 slices. Report confidence intervals (bootstrap by season) because single-season differences are noise-dominated.
- [ ] `src/sports_ml/sports/football/models/training.py` & `artifacts/football/`:
  - Model bundle persistence with `metadata.json` SHA-256 schema fingerprints consistent with other sports.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_stage1_model.py`:
  - Verify $P(\text{Home}) \in (0,1)$, $P(\text{Away}) = 1 - P(\text{Home})$ (within `1e-9`), `prob_draw == 0.0`, finite non-negative expected scores, and neutral-site games are symmetric under home/away relabeling (within tolerance).
  - Verify tie games are excluded from classifier fit/calibration but included in Elo updates at `0.5`.
  - Verify walk-forward ordering (`gbdt_train_max_time < calibrator_min_time <= calibrator_max_time < test_min_time`) and that no fold trains on future weeks.
  - Verify calibrated models beat the empirical home-win prior and Elo-only baseline on a synthetic directional dataset, and that the calibration-size guard triggers the identity fallback.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 26: Stage 2 QB / RB / WR-TE Projections, Anytime-TD Probabilities, Unified Football Predictor & CLI (`sports/football/models/` & `cli.py`)
**Status**: `[ ] Not Started`

### Goal
Implement optional Stage 2 player projections conditioned on Stage 1 (QB, RB, WR/TE; **anytime-TD** and milestone probabilities), wire `FootballMatchupPredictor` into the `sports-ml` CLI with week-based addressing, and update skills/docs.

### Deliverables
- [ ] `src/sports_ml/sports/football/features/player.py` & `src/sports_ml/sports/football/models/stage2_player.py`:
  - **Player feature matrix** (leak-free `.shift(1)`, shrunk per-opportunity rates): QB attempts/air yards/`cpoe`; RB `carry_share`, `target_share`, goal-line share, snap share; WR/TE `target_share`, `air_yards_share`, `wopr`, `route_participation`, red-zone target share; opposing defense vs. position EPA/yards allowed; rookie/debut priors from draft capital; availability probability from Tier 1/2 (`p_active`).
  - **Team Opportunity Model**: Projects team `proj_plays`, `proj_pass_attempts`, `proj_rush_attempts` from Stage 1 expected scores/margin (game-script effect: trailing teams pass more), pace, and pass-rate-over-expected.
  - **QB**: `proj_pass_att`, `proj_completions`, `proj_pass_yds`, `proj_pass_td`, `proj_int`, `proj_rush_yds`, `proj_rush_td`, `prob_250plus_pass_yds`, `prob_300plus_pass_yds`, `prob_2plus_pass_td`.
  - **RB**: `proj_carries`, `proj_rush_yds`, `proj_rush_td`, `proj_targets`, `proj_rec`, `proj_rec_yds`, `prob_100plus_rush_yds`.
  - **WR/TE**: `proj_targets`, `proj_rec`, `proj_rec_yds`, `proj_rec_td`, `prob_5plus_rec`, `prob_75plus_rec_yds`, `prob_100plus_rec_yds`.
  - **Anytime TD (`prob_anytime_td`)** for RB/WR/TE/QB (rushing + receiving) from zero-inflated / Poisson TD-rate models tied to red-zone usage and team expected points.
  - **Cross-Stage Coherence Reconciler (`reconcile_football_stage2_projections`)**: Soft-scales player TD and yardage totals toward team expected points / projected passing & rushing totals ($\gamma = 0.50$), and enforces opportunity conservation (sum of projected targets $\approx$ team projected attempts; carries sum $\approx$ team rush attempts).
- [ ] `src/sports_ml/sports/football/models/predictor.py`:
  - `FootballMatchupPredictor` with `predict_matchup(season, week, away, home, include_players=False)`, `predict_week(season, week, include_players=False)`, `predict_date(...)`, and `predict_team_next_games(team, n_games, include_players=False)`; appends to `data/predictions/football_predictions.parquet`.
  - Confidence warnings: `CONFIRMED_QB` vs. `PROBABLE_QB` vs. `QB_UNCERTAIN` (questionable/doubtful) vs. `BACKUP_QB`, `EARLY_SEASON_SHRINKAGE` (Weeks 1–4), `DEBUT_OR_NEW_QB_PRIOR`, `NEUTRAL_OR_INTERNATIONAL_SITE`, `KEY_OFFENSIVE_PLAYER_OUT`, `STAGE1_ONLY_NO_PLAYER_DATA`.
- [ ] `src/sports_ml/cli.py`:
  - Register `--sport football` (with `--league nfl`, the default) across `check-sources`, `ingest`, `features build`, `train`, `backtest`, `predict`, and `score-predictions`, adding `--season` and `--week` options (and `--matchup AWAY@HOME`, `--team`, `--games`, `--include-players` consistent with other sports); extend `.github/skills/*` (ingest-data, train-models, predict-matchup, evaluate-and-score, onboard-sport-plugin) and docs for NFL.
  - Rich output: `P(Away)`, `P(Home)`, projected score line, QB matchup card (`Starter`, `Confirmed/Probable`, `QB Value`), and position-grouped player projection tables including `P(Anytime TD)`.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_stage2_player.py`:
  - Verify all projected counts are non-negative, probabilities lie in $[0,1]$, `prob_100plus_*` $\le$ `prob_75plus_*`, and a player with `p_active = 0` projects `0` opportunities.
  - **Opportunity Conservation**: projected target/carry/attempt sums reconcile to team totals within tolerance.
  - **Game-Script Monotonicity**: Raising Stage 1 team expected margin (leading) increases RB `proj_carries` and decreases QB `proj_pass_att`; raising team $\hat{\mu}$ increases anytime-TD probabilities.
  - **Role Monotonicity**: a higher `target_share` WR has strictly higher `proj_targets` than an otherwise identical lower-share WR.
- [ ] `tests/sports/football/test_football_cli_e2e.py`: Offline end-to-end `ingest` $\rightarrow$ `features build` $\rightarrow$ `train` $\rightarrow$ `backtest` $\rightarrow$ `predict` (Stage 1, `--include-players`, `--week`, `--team`) $\rightarrow$ `score-predictions`.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 27: Weather, Roof/Surface & Injury / Depth-Chart Availability Features (`sports/football/ingestion/` & `features/`)
**Status**: `[ ] Not Started`

### Goal
Add environment and roster-availability signals that matter in the NFL (wind/precipitation/cold for passing and kicking; missing OL, pass rush, secondary, and skill players) while preventing **observed-weather** and **post-game inactive** leakage.

### Deliverables
- [ ] `src/sports_ml/sports/football/ingestion/weather.py`:
  - Pre-game **forecast** weather per outdoor/open-roof venue (`forecast_temp_f`, `forecast_wind_mph`, `forecast_precip_prob`, `forecast_lead_hours`) from a cached historical-forecast source (e.g., Open-Meteo historical forecast archive) snapshotted at a fixed lead time before kickoff; domes and closed-roof games get neutral indoor values. Observed nflverse `temp`/`wind` are permitted only as a flagged fallback (`weather_is_observed_proxy = 1`) for seasons without forecast coverage and are never silently mixed with forecasts.
- [ ] `src/sports_ml/sports/football/features/environment.py`:
  - `wind_penalty` (nonlinear above ~15 mph), `cold_penalty`, `is_precipitation_risk`, roof/surface/altitude interactions, team-specific dome/warm-weather-team-in-cold flags, and expected-points environment shifts (feeding Stage 1 totals and Stage 2 passing/kicking projections).
- [ ] `src/sports_ml/sports/football/features/availability.py`:
  - Position-group availability from Tier 1/Tier 2 snapshots weighted by prior snap share / player value: `ol_availability_index`, `pass_rush_availability_index`, `secondary_availability_index`, `skill_availability_index`, `starter_snap_share_missing`, `qb_p_starts`, with `diff_*` features and as-of-time stamps.
  - Injury-report designations are converted to **historical start probabilities** (e.g., empirical $P(\text{plays} \mid \text{Questionable})$ estimated only from games $<$ the prediction date) rather than hard 0/1 assumptions.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_football_weather_availability.py`:
  - Verify a high-wind forecast lowers expected passing environment monotonically and has no effect in domes/closed-roof games; verify forecast-vs-observed columns are never mixed and the observed-proxy flag is set when used.
  - **Strict As-Of Test**: mutating inactives/injury designations stamped after the as-of time leaves features bit-for-bit unchanged; mutating post-kickoff weather leaves forecast features unchanged.
  - Verify losing a starting LT / top pass rusher decreases the corresponding availability index proportionally to prior snap share, and that empirical `Questionable` start probabilities are estimated without future games.
- [ ] Run `sports-ml backtest --sport football --league nfl` before/after to confirm Stage 1 Log Loss / score MAE do not regress, reporting the delta with season-bootstrap confidence intervals.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 28: Play-by-Play Micro-Features: Pressure, CPOE, Pace, PROE & Situational Efficiency (`sports/football/ingestion/` & `features/`)
**Status**: `[ ] Not Started`

### Goal
Compress play-level nflverse data and Next Gen Stats into compact, high-signal pre-game features beyond coarse EPA, and wire them into Stage 1 and Stage 2.

### Deliverables
- [ ] `src/sports_ml/sports/football/ingestion/pbp_micro.py` & `ngs.py`:
  - Aggregate play-level data into `zstd` Parquet `data/silver/football/team_pbp_micro_logs.parquet` and `player_pbp_micro_logs.parquet`:
    - **Pace & Script**: neutral-situation seconds/play, plays per game, **Pass Rate Over Expected (`proe`)** (neutral game-state), 4th-down aggressiveness vs. model (`fourth_down_go_rate_over_expected`), two-minute/late-game efficiency.
    - **Pass Game Quality**: `cpoe`, `air_yards_per_attempt`, `adot`, `yac_over_expected`, `pressure_rate_allowed`/`pressure_rate_generated` (where available, e.g., from Next Gen Stats / PFR-type fields), `sack_rate_over_expected`, `time_to_throw`.
    - **Situational**: `red_zone_epa_per_play`, `third_down_epa`, `explosive_pass_rate_allowed`, `rush_yards_over_expected` (NGS), `run_stuff_rate`, penalty-adjusted EPA, and score-state-neutral (win-probability banded) splits.
    - **Player micro**: `target_share`, `air_yards_share`, `yards_per_route_run` (when routes are available), `separation`, `rush_yards_over_expected`, `efficiency` vs. usage.
  - Respect source availability windows (`has_ngs = False` before 2016) with graceful, flagged imputation.
- [ ] `src/sports_ml/sports/football/features/pipeline.py`:
  - Wire rolling `.shift(1)` micro-features into Stage 1 (`diff_proe`, `diff_pressure_rate_edge`, `diff_neutral_pace`, `diff_red_zone_epa`, `diff_cpoe_vs_coverage`) and Stage 2 (player rolling `yprr`, `rush_yards_over_expected`, `air_yards_share`), preserving the Milestone 24 odds/observed-outcome firewall.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_football_micro_features.py`:
  - **Strict Temporal Anti-Leakage Test**: mutating play-level data at games $\ge t$ leaves all pre-game micro-features for games $< t$ bit-for-bit unchanged.
  - **Conservation Test**: play-level EPA aggregated by team-game equals the team-game EPA total in `team_game_logs.parquet` (within `1e-6`), and neutral-situation filters never include plays outside the configured score/time band.
  - **Availability Test**: Pre-2016 seasons produce finite imputed micro-features with `*_is_imputed = 1` and zero `NaN`s.
- [ ] Re-run `sports-ml backtest --sport football --league nfl` to verify out-of-sample Log Loss / Brier / score MAE improvement (reported with season-bootstrap CIs; ship the micro-feature set only if it does not regress).
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

## Milestone 29: NFL Closing-Odds Benchmark, De-Vigging & ATS / Total Evaluation Firewall (`sports/football/ingestion/odds.py`)
**Status**: `[ ] Not Started`

### Goal
Benchmark NFL Stage 1 against closing market lines (moneyline-implied probabilities, closing spread, closing total) under the **strict `market_*` anti-contamination firewall** established in Milestone 13—market data is used for **evaluation only**, never as a model input.

### Deliverables
- [ ] `src/sports_ml/sports/football/ingestion/odds.py`:
  - `FootballOddsIngestor`: Reads closing moneyline, spread, and total from the quarantined nflverse schedule columns (Milestone 23) plus an optional local historical-odds CSV (`data/raw/odds/nfl_historical_odds.csv.gz`) and optional `THE_ODDS_API_KEY` pre-game snapshot (gracefully skipped when absent). Persists nullable `market_home_decimal_odds`, `market_away_decimal_odds`, `market_prob_home`, `market_prob_away` (power de-vigged via `core/market.py`), `market_spread_home`, and `market_total_points` to `data/silver/football/games.parquet`.
- [ ] `src/sports_ml/core/market.py` & `core/evaluation.py` (additive):
  - Spread/total evaluators: model-implied margin vs. closing spread (**ATS** hit rate with push handling, expected-value at standard $-110$ juice, break-even $52.38\%$), model-implied total vs. closing total (**over/under** hit rate), and **closing-line value (CLV)** summaries, bucketed by disagreement $\tau \in \{1, 2, 3\}$ points, with explicit **key-number (3, 6, 7, 10)** handling for pushes.
- [ ] `src/sports_ml/cli.py`:
  - Display the **De-Vigged Market Closing Line** benchmark and the **Fundamental vs. Market Disagreement** / ATS / total breakdown in `sports-ml backtest --sport football --league nfl` and `score-predictions --sport football --league nfl` when odds are present.

### Verification & Acceptance Criteria
- [ ] `tests/sports/football/test_football_odds_firewall.py` (extended) & `tests/core/test_market.py` (extended):
  - **Strict Anti-Contamination Test**: no `market_*`, `*odds*`, or `spread*`/`total*` market column in `STAGE1_FEATURE_COLUMNS` / `STAGE2_FEATURE_COLUMNS`; mutating `market_*` columns changes `0.0` of predicted probabilities and expected scores while updating the `MarketEdgeReport`.
  - Verify de-vigged 2-way probabilities lie strictly in $(0,1)$ and sum to `1.0` (within `1e-9`); ATS evaluator correctly handles pushes, `-110` break-even math, and half-point vs. whole-point spreads.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.

---

# Phase 7: User Personalization

## Milestone 30: Starred Teams & Cross-Sport `predict --starred` (`config/`, `core/starred.py` & `cli.py`)
**Status**: `[ ] Not Started`

### Goal
Let the user maintain a personal list of favorite teams across every supported sport and run a single command that shows the calibrated predictions for each starred team's next game(s). This is a thin orchestration layer over each plugin's existing `predict_team_next_games(team, n_games, include_players)`; it adds **no** new modeling and must not change any prediction output.

### Design Constraints
1. **Teams are keyed by `(sport, league, team)`**, never a bare tricode (`PHI`, `BOS`, `SEA`, `TOR`, `LAC`, etc. collide across sports). `league` is `"nhl"`, `"epl"`, `"mlb"`/`"npb"`, or `"nfl"`; baseball entries may omit the league and rely on the predictor's existing MLB-vs-NPB auto-resolution, but the resolved league is stored on add.
2. **Sport-agnostic core, sport-specific resolution**: `core/` only stores/validates entries and orchestrates calls; each sport adapter supplies team validation/normalization (reusing existing alias maps, e.g., `OAK`→`ATH`, `WSH`/`WAS`) via a small `TeamResolver` protocol, so new sports (e.g., football) register by exposing a resolver and a predictor.
3. **Graceful partial failure**: one starred team being out of season, having no scheduled game, or belonging to a sport with no trained artifacts/stale Silver data must never abort the command; it produces a clear per-team `SKIPPED` line (`NO_UPCOMING_GAME`, `NO_TRAINED_ARTIFACTS`, `SOURCE_UNAVAILABLE`) and the remaining teams still render. Exit code is `0` if at least one prediction rendered, non-zero only if every entry failed.
4. **Sync once per sport, not per team**: live ingestion refresh (unless `--offline`) runs at most once per sport per invocation.
5. **No prediction-semantics drift**: starred runs reuse the exact existing predictor methods and ledger append path, so predictions are identical to `predict --sport X --team Y --games N`.

### Deliverables
- [ ] `src/sports_ml/core/starred.py`:
  - `StarredTeam` (frozen pydantic model: `sport`, `league`, `team`, optional `label`) and `StarredTeamsConfig` with load/save/add/remove/list, deduplication on `(sport, league, team)`, schema validation with a clear error for unknown sports, and atomic writes.
  - `TeamResolver` protocol (`resolve(team: str, league: str | None) -> ResolvedTeam`) and a registry reusing `core/registry.py`; implementations added for hockey, soccer, and baseball (and registered by the football plugin once Milestone 22 lands; football must not be a prerequisite for this milestone).
- [ ] `config/starred_teams.toml` (git-ignored; path overridable via `SPORTS_ML_STARRED_TEAMS_FILE` and honoring `SPORTS_ML_ROOT_DIR`), plus a committed `config/starred_teams.example.toml`; add `config/starred_teams.toml` to `.gitignore`. A missing file is treated as an empty list (not an error).
- [ ] `src/sports_ml/cli.py`:
  - `sports-ml starred add --sport hockey --team COL`, `starred add --sport baseball --team LAD` (league auto-resolved), `starred remove ...`, `starred list` (Rich table of sport / league / team / full name).
  - `sports-ml predict --starred [--games N] [--include-players] [--offline]` and the alias `sports-ml predict starred`, with `--games` defaulting to `1` (next game per team). `--starred` is mutually exclusive with `--sport`, `--team`, `--matchup`, and `--date` (clear error message) and, when combined with `--include-players`, applies to every sport that supports Stage 2.
  - Orchestrator groups entries by sport, builds one predictor per sport, calls `predict_team_next_games(...)` per team, merges results into **one chronologically sorted view by kickoff time** (displayed in the user's local timezone, with the sport/league shown per row), and delegates rendering to the existing per-sport renderers (2-way vs. 3-way tables stay separate per sport; no new table formats).
  - Empty list prints a helpful hint (`sports-ml starred add --sport ... --team ...`) and exits `0`.
- [ ] Update `README.md` (CLI examples), `docs/architecture.md` (CLI section), and the `predict-matchup` skill playbook with the `starred` workflow.

### Verification & Acceptance Criteria
- [ ] `tests/core/test_starred.py`:
  - Verify add/remove/list round-trips, deduplication on `(sport, league, team)`, that `PHI` in hockey and `PHI` in baseball coexist as separate entries, alias normalization (e.g., `OAK` stored as `ATH`), rejection of unknown sports/teams, atomic write (no partial file on failure), missing file $\rightarrow$ empty list, and `SPORTS_ML_STARRED_TEAMS_FILE` / `SPORTS_ML_ROOT_DIR` overrides.
- [ ] `tests/test_starred_cli_e2e.py` (offline fixtures, stubbed/trained tiny artifacts):
  - Starring one team in each supported sport and running `predict --starred` returns each team's next game in chronological order with the same probabilities as `predict --sport X --team Y --games 1` (bit-for-bit identical).
  - **Partial-failure Test**: a starred team with no upcoming game and a sport with missing artifacts produce `SKIPPED` rows while other teams still render and exit code is `0`; all-failed returns non-zero.
  - **Sync-Once Test**: with the network layer mocked, two starred teams in the same sport trigger a single ingestion refresh.
  - `predict --starred` combined with `--sport`/`--team`/`--matchup`/`--date` fails with a clear message, and `predict starred` behaves identically to `predict --starred`.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest --cov=sports_ml` pass with 0 errors.




