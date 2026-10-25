# System Architecture Blueprint

> **Related Documents**: [Data & Feature Specification](./data-and-features.md) | [Implementation Roadmap](./roadmap.md)

## 1. Executive Summary & Design Goals

`sports-analytics-ml` is a modular, leak-free machine learning platform built in **Python 3.11+** using **Polars** and **PyArrow**. Starting with **NHL Hockey**, the system ingests historical and daily schedule, roster, play-by-play/shot-quality, and goaltending data from the **Official NHL Web API** and **MoneyPuck**, engineers strictly pre-game temporal features, and executes a **Two-Stage Prediction Pipeline**:

1. **Stage 1 — Game Outcome Model (Mandatory)**: Produces calibrated win probabilities $P(\text{Home Win})$ and $P(\text{Away Win}) = 1 - P(\text{Home Win})$, alongside expected team scoring pace context, evaluated via walk-forward time-series backtesting (**Log Loss**, **Brier Score**, and Expected Calibration Error).
2. **Stage 2 — Player Matchup Stat Projections (Optional)**: On-demand per-player predictions for both teams' active rosters in a matchup (Skaters: expected goals, assists, points, shots on goal, TOI, and point/goal probabilities; Goalies: expected shots against, saves, goals against, save %), conditioned on Stage 1 game environment and opponent strength.

### Core Architectural Tenets
* **Multi-Sport Extensibility**: Sport-agnostic time-series engines (Elo/Glicko ratings, rolling/EWMA transformers, rest/travel schedule calculators, walk-forward splitters, calibration wrappers) live in `src/sports_ml/core/`. Sport-specific ingestion, domain metrics, and roster rules live in `src/sports_ml/sports/<sport>/` (starting with `hockey/`).
* **Zero Data Leakage by Construction**: All feature pipelines enforce a strict temporal firewall—any feature generated for game $g$ starting at timestamp $T_g$ is computed exclusively from games completed at $T < T_g$.
* **Dual Interface (CLI + Python Package)**: Every workflow (ingestion, feature generation, training, backtesting, and matchup prediction) is exposed both as a typed Python API and via the `sports-ml` CLI.

---

## 2. High-Level System Data Flow

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL DATA SOURCES                            │
│  • Official NHL Web API (api-web.nhle.com): Schedules, Rosters, Boxscores  │
│  • MoneyPuck CSV Snapshots: 5v5 & All-Situation xG, Corsi, Fenwick, GSAx   │
│  • Pre-Game Lineup & Goalie Feeds (DailyFaceoff): Morning-skate starters,  │
│    projected line combinations, and day-to-day injury/scratch status       │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & STORAGE LAYER (Parquet Medallion Architecture)              │
│  • data/raw/     -> Immutable HTTP JSON / CSV caches (hash & date keyed)   │
│  • data/bronze/  -> Normalized Polars schemas (games, team_games, players) │
│  • data/silver/  -> Joined canonical pre-game & post-game entity logs      │
│  • data/gold/    -> Leak-free feature matrices (game_features, player_feat)│
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. LEAK-FREE FEATURE ENGINE (Polars LazyFrames + Temporal Shift Guard)     │
│  ┌──────────────────────────────────┐  ┌─────────────────────────────────┐ │
│  │ Core Agnostic Transformers       │  │ Hockey Plugin Transformers      │ │
│  │ • Dynamic Elo (HFA, MOV, Regress)│  │ • 5v5 xGF%, CF%, FF%, HD Share  │ │
│  │ • Left-closed Rolling & EWMA     │  │ • Special Teams (PP%, PK% edge) │ │
│  │ • Rest Days, B2B, 3-in-4, Travel │  │ • PDO Luck / Regression Signal  │ │
│  │ • Season-Boundary Shrinkage      │  │ • Starting Goalie Rolling GSAx  │ │
│  └──────────────────────────────────┘  │ • TOI-Weighted Roster Strength  │ │
│                                        └─────────────────────────────────┘ │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. TWO-STAGE PREDICTION PIPELINE                                           │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: MANDATORY GAME OUTCOME MODEL                                │  │
│  │ • Inputs: Differential & Raw Pre-Game Team, Elo, Goalie, Roster Feat │  │
│  │ • Estimators: Elo Baseline + LightGBM/XGBoost + Probability Calib    │  │
│  │ • Output: Calibrated P(Home Win), P(Away Win), Implied Pace/Env      │  │
│  └──────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                      │
│                                     ▼ (Optional: --include-players)        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: OPTIONAL PLAYER MATCHUP STAT PROJECTION MODEL               │  │
│  │ • Inputs: Player Rolling Rates (per-60 & per-game) + Projected TOI   │  │
│  │           + Opponent Defensive xGA/60 &PK + Opponent Goalie GSAx     │  │
│  │           + Stage 1 Context (P(Win), Implied Team Scoring Pace)      │  │
│  │ • Estimators: Skater & Goalie Multi-Target Count/Rate Regressors     │  │
│  │ • Reconciliation: Coherent scaling with Stage 1 team environment     │  │
│  │ • Output: Skater (E[G], E[A], E[PTS], E[SOG], E[TOI], P(1+ PT/G))    │  │
│  │           Goalie (E[SA], E[SV], E[GA], E[SV%])                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Two-Stage Prediction Architecture

A core requirement of the system is a **Two-Stage Prediction Process** during inference and evaluation:

### Stage 1: Calibrated Game Win Probability (Mandatory)
Every matchup prediction runs Stage 1.
* **Primary Target**: Binary outcome $y_{\text{win}} \in \{0, 1\}$, where $1$ denotes a Home Team win (including Overtime and Shootout) and $0$ denotes an Away Team win.
* **Auxiliary Context Target (Internal to Stage 1)**: Expected regulation/total team scoring rate $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$ to quantify the matchup's offensive environment for Stage 2 conditioning.
* **Model Ensemble & Calibration**:
  1. **Dynamic Elo Baseline**: Produces an analytical win probability $P_{\text{Elo}}(\text{Home Win})$ from pre-game team ratings, home-ice advantage, and rest differentials.
  2. **Gradient Boosted Decision Trees (LightGBM / XGBoost)**: Trained on the full pre-game feature vector (Elo, rolling 5v5 $xG\%$, Corsi/Fenwick, special teams, PDO, starting goalie GSAx, TOI-weighted active roster quality, and schedule fatigue).
  3. **Out-of-Fold Probability Calibration**: Sigmoid (Platt scaling) or Isotonic Regression fitted strictly on a held-out chronological validation fold so predicted probabilities reflect true empirical win frequencies.

### Stage 2: Player Matchup Stat Projections (Optional)
When the user requests player projections (e.g., `sports-ml predict --home BOS --away TOR --date 2026-10-25 --include-players` or `predictor.predict_matchup(..., include_players=True)`), the pipeline executes Stage 2 for all active roster skaters and starting goalies in the matchup.

* **Why Condition Stage 2 on Stage 1?**
  Individual player counting stats depend heavily on both **individual usage/skill** and the **macro game environment** (matchup pace, opponent defensive suppression, opponent starting goalie, and game script/win probability). Passing Stage 1's calibrated $P(\text{Home Win})$ and expected team scoring environment into Stage 2 ensures player projections are context-aware.
* **Stage 2 Sub-Models**:
  1. **Usage / TOI Projector**: Predicts each active skater's expected Time on Ice ($\widehat{\text{TOI}}_{\text{even}}$, $\widehat{\text{TOI}}_{\text{PP}}$) based on their rolling deployment and any missing/injured teammates on the active roster.
  2. **Skater Stat Models**:
     * Predicts expected rate/count targets using Poisson / Tweedie objective LightGBM regressors:
       * $\hat{\mu}_{\text{SOG}}$ (Expected Shots on Goal)
       * $\hat{\mu}_{\text{G}}$ (Expected Goals)
       * $\hat{\mu}_{\text{A}}$ (Expected Assists)
       * $\hat{\mu}_{\text{PTS}} = \hat{\mu}_{\text{G}} + \hat{\mu}_{\text{A}}$ (Expected Points)
     * Derives milestone probabilities via Poisson survival functions:
       * $P(\text{Goals} \ge 1) = 1 - \exp(-\hat{\mu}_{\text{G}})$
       * $P(\text{Points} \ge 1) = 1 - \exp(-\hat{\mu}_{\text{PTS}})$
  3. **Goalie Stat Model**:
     * Predicts $\hat{\mu}_{\text{SA}}$ (Expected Shots Against), $\hat{\mu}_{\text{GA}}$ (Expected Goals Against), $\hat{\mu}_{\text{SV}} = \hat{\mu}_{\text{SA}} - \hat{\mu}_{\text{GA}}$, and expected Save Percentage $\widehat{\text{SV\%}}$.
* **Hierarchical Coherence Reconciliation**:
  * To prevent the sum of individual skater goal projections from drifting wildly from the team's macro scoring expectation, Stage 2 applies an optional soft proportional scaling factor so $\sum_{i \in \text{Roster}} \hat{\mu}_{G, i}$ aligns with the team's expected regulation/OT goals.

---

## 4. Multi-Sport Plugin Contract (NHL, MLB, PGA Extensibility)

To ensure adding future sports—specifically **Baseball (MLB)** and **Golf (PGA)**—requires **zero rewrites** of `src/sports_ml/core/`, the core engine is designed around two competition topologies (`head_to_head` and `multi_entrant_field`) and generalized exposure-normalized temporal primitives:

### 4.1 How `core/` Generalizes Across Hockey, Baseball, and Golf

| Architectural Dimension | `core/` Abstraction | Hockey (`sports/hockey/`) | Baseball (`sports/baseball/` - Future) | Golf (`sports/golf/` - Future) |
| :--- | :--- | :--- | :--- | :--- |
| **Competition Topology** | `head_to_head` (2 sides) & `multi_entrant_field` ($N$ entrants in an `event_id`) | `head_to_head` (Home Team vs. Away Team) | `head_to_head` (Home Team vs. Away Team, incl. same-day doubleheaders) | Both `multi_entrant_field` (120–156 golfers in 4-round event) & `head_to_head` (tournament/round matchups) |
| **Temporal Ordering Key** | `(start_time_utc, event_seq, event_id)` (never assumes 1 game/day) | Regular season / playoff game timestamp | Handles Game 1 (`event_seq=1`) vs. Game 2 (`event_seq=2`) of MLB doubleheaders | Handles Round 1–4 (`event_seq=1..4`) sequential cut & tee-time ordering |
| **Exposure Rate Basis** | `pregame_exposure_rate(num, denom, scale)` | Per 60 minutes of TOI (`scale=60.0`) | Per PA / BF (`scale=1.0`) for batters; per 9 IP (`scale=9.0`) for pitchers | Per 18-hole round (`scale=1.0` or `18.0`) for Strokes Gained categories |
| **Key Specialist / Context Unit** | Pluggable `ContextModifier` & `StarterUnit` hooks | Starting Goalie GSAx + Arena coordinates | Starting Pitcher (FIP/xFIP/Stuff+) + Bullpen workload + Ballpark/Weather factors | Course-fit profile (driving distance/accuracy, grass type) + Field strength |
| **Stage 1 Mandatory Output** | Calibrated event outcome probabilities | Binary $P(\text{Home Win})$ + implied team goals $(\hat{\lambda}_H, \hat{\lambda}_A)$ | Binary $P(\text{Home Win})$ + implied team runs $(\hat{\lambda}_H, \hat{\lambda}_A)$ | Group-normalized field probs $\sum_i P(\text{Win}_i)=1$, $P(\text{Top 5/10/20})$, $P(\text{Make Cut})$, or H2H $P(A \text{ beats } B)$ |
| **Stage 2 Optional Output** | Conditioned participant stat projections | Skaters: $E[\text{G, A, PTS, SOG, TOI}]$; Goalies: $E[\text{SA, SV, GA}]$ | Batters: $E[\text{H, HR, TB, RBI, R, K}]$; Pitchers: $E[\text{IP, K, ER, Outs}]$ | Golfers: $E[\text{SG:OTT, SG:APP, SG:ATG, SG:PUTT, Birdies, GIR, Score}]$ |

### 4.2 Core vs. Sport Plugin Responsibilities

| Component | `core/` Responsibility (Sport-Agnostic) | `sports/<sport>/` Responsibility (Sport Plugin) |
| :--- | :--- | :--- |
| **Schemas (`contracts.py`)** | Enforces canonical event keys `(sport, season, event_id, event_seq, start_time_utc)` and participant keys `(event_id, entity_id, participant_id)`. | Maps sport API IDs (NHL, MLB StatsAPI/Statcast, PGA/DataGolf) into canonical schemas. |
| **Ratings (`ratings.py`)** | Pairwise Elo/Glicko engine (with starter-adjustment hooks) AND multi-entrant Plackett-Luce / field-adjusted skill rating engine. | Configures sport parameters (e.g., NHL $K=6$ & OT dampening; MLB SP-adjusted Elo; PGA field-adjusted Strokes Gained skill). |
| **Rolling Engine (`temporal.py`)** | Strictly shifted (`shift(1)`) Polars window expressions over $N$ appearances or EWMA spans $\alpha$, generalized `pregame_exposure_rate`, and Bayesian shrinkage. | Defines sport-specific columns to roll ($xGF_{5v5}$ & GSAx in NHL; $xwOBA$ & $xFIP$ in MLB; $SG:Total$ splits in PGA). |
| **Schedule (`schedule.py`)** | Computes days rest, back-to-backs, doubleheader turnarounds, $k$-in-$m$ density, geodesic venue travel distance, and timezone shifts. | Supplies venue coordinates, timezones, and optional venue/park/course environment factors. |
| **Modeling & Eval** | Walk-forward splitter, GBDT trainers, binary & group-softmax probability calibrators, Log Loss / Brier / Ranking / Poisson deviance evaluators. | Registers Stage 1 (Event Outcome) and Stage 2 (Participant Stat) feature sets and target columns. |

---

## 5. Repository Directory Structure

```text
sports-analytics-ml/
├── .github/
│   ├── agents/
│   │   ├── sports-ml-architect.agent.md
│   │   └── sports-ml-developer.agent.md
│   ├── skills/
│   │   ├── evaluate-and-score/
│   │   │   └── SKILL.md             # /evaluate-and-score backtest & live ledger scoring
│   │   ├── implement-milestone/
│   │   │   └── SKILL.md             # /implement-milestone workflow & quality gate
│   │   ├── ingest-data/
│   │   │   └── SKILL.md             # /ingest-data NHL API & MoneyPuck Silver sync
│   │   ├── leak-free-feature-audit/
│   │   │   └── SKILL.md             # /leak-free-feature-audit rules & pytest template
│   │   ├── nhl-data-integration/
│   │   │   └── SKILL.md             # NHL Web API v1, MoneyPuck & 2-Tier lineup specs
│   │   ├── onboard-sport-plugin/
│   │   │   └── SKILL.md             # Playbook for adding MLB Baseball & PGA Golf plugins
│   │   ├── predict-matchup/
│   │   │   └── SKILL.md             # /predict-matchup Stage 1 & Stage 2 CLI workflow
│   │   └── train-models/
│   │       └── SKILL.md             # /train-models Gold feature build & model bundle fit
│   └── workflows/
│       └── ci.yml                   # Fast offline CI gate (ruff, mypy, pytest)
├── docs/
│   ├── architecture.md              # This system architecture blueprint
│   ├── data-and-features.md         # Schemas, data sources & mathematical feature specs
│   └── roadmap.md                   # Phased implementation milestones & test gates
├── data/                            # Git-ignored local data lakehouse (Parquet + JSON cache)
│   ├── raw/                         # Cached NHL API JSON responses & MoneyPuck CSVs
│   ├── bronze/                      # Normalized typed Parquet tables
│   ├── silver/                      # Entity-aligned game, team_game, skater_game, goalie_game
│   ├── gold/                        # Pre-game feature matrices for Stage 1 & Stage 2
│   └── predictions/                 # Append-only live prediction ledger (Parquet audit log)
├── artifacts/                       # Git-ignored trained model bundles + metadata.json fingerprints
├── .gitignore                       # Blocks data/, artifacts/, .venv/, and caches from git
├── pyproject.toml                   # Dependencies, ruff, mypy, and pytest configuration
├── uv.lock                          # Reproducible transitive dependency lockfile
├── src/
│   └── sports_ml/
│       ├── __init__.py
│       ├── cli.py                   # Typer/Click CLI (`sports-ml ingest|features|train|backtest|predict`)
│       ├── config.py                # Pydantic settings (paths, seasons, hyperparameters)
│       ├── core/                    # SPORT-AGNOSTIC ML ENGINE
│       │   ├── __init__.py
│       │   ├── contracts.py         # SportAdapter protocol & Pydantic/Polars schema contracts
│       │   ├── temporal.py          # Anti-leakage guards, shift(1) rolling & EWMA Polars expressions
│       │   ├── ratings.py           # Dynamic Elo / Glicko rating engine
│       │   ├── schedule.py          # Rest days, B2B, 3-in-4, arena haversine travel & timezone calculator
│       │   ├── shrinkage.py         # Early-season Bayesian prior / prior-season blending
│       │   ├── validation.py        # Walk-forward expanding & sliding window time-series splitters
│       │   ├── calibration.py       # Out-of-fold Platt / Isotonic probability calibration
│       │   ├── evaluation.py        # Log Loss, Brier Score, ECE, ROI & player MAE/Poisson metrics
│       │   └── registry.py          # Artifact serialization, metadata & model registry
│       └── sports/
│           └── hockey/              # NHL HOCKEY PLUGIN
│               ├── __init__.py
│               ├── adapter.py       # HockeyAdapter implementing SportAdapter
│               ├── constants.py     # NHL 32 teams, arena coordinates, ID mappings
│               ├── ingestion/
│               │   ├── nhl_api.py   # Async/cached client for api-web.nhle.com (schedule, boxscore, roster)
│               │   ├── moneypuck.py # Downloader & parser for MoneyPuck teams/skaters/goalies CSVs
│               │   ├── lineups.py   # 2-Tier live pre-game lineup, starting goalie & injury resolver
│               │   └── reconciler.py# Joins NHL API & MoneyPuck into Bronze/Silver Parquet tables
│               ├── features/
│               │   ├── team.py      # 5v5 xG%, Corsi, Fenwick, PDO, Special Teams rolling features
│               │   ├── goalie.py    # Starting goalie resolver & rolling GSAx / SV% features
│               │   ├── roster.py    # TOI-weighted active skater aggregation for Stage 1 game features
│               │   └── player.py    # Individual skater & goalie matchup feature builder for Stage 2
│               └── models/
│                   ├── stage1_game.py   # Mandatory Calibrated Win Probability model pipeline
│                   ├── stage2_player.py # Optional Skater/Goalie stat projection pipeline & reconciler
│                   └── predictor.py     # Unified two-stage matchup inference service
└── tests/
    ├── conftest.py                  # Synthetic multi-season NHL fixtures (deterministic Polars DataFrames)
    ├── core/
    │   ├── test_temporal_leakage.py # Property tests proving future mutations never alter past features
    │   ├── test_ratings.py          # Elo update, HFA, OT handling & offseason regression tests
    │   ├── test_schedule.py         # B2B, rest, travel distance & boundary tests
    │   └── test_validation.py       # Walk-forward chronological split tests
    └── sports/
        └── hockey/
            ├── test_ingestion.py    # Schema normalization & ID reconciliation tests
            ├── test_hockey_features.py # xG%, PDO, goalie GSAx & TOI-weighted roster tests
            ├── test_stage1_model.py # Calibrated win probability bounds & sum-to-one tests
            └── test_stage2_player.py# Optional player stat projection & stage-coherence tests
```

---

## 6. CLI & Python API Ergonomics

### Command-Line Interface (`sports-ml`)
Built with `typer` and `rich`. The CLI is designed around two workflows: a **streamlined daily workflow** (where intermediate steps run automatically under the hood) and **granular developer commands** for data backfills, feature experimentation, and walk-forward accuracy testing.

#### A. Streamlined Daily & Setup Workflow
You do **not** need to manually run `features build` or `backtest` just to train a model or predict tonight's games:
* `sports-ml train` automatically builds/updates the Gold feature tables from your Silver data if they are missing or stale before fitting the model.
* `sports-ml predict` **connects live by default** (use `--offline` only if you want to skip network calls): it incrementally syncs completed current-season games through yesterday, pulls tonight's live matchup context via the **2-Tier Live Lineup & Goalie Resolver** (Tier 1: NHL API official pre-game warmup lineup & active roster; Tier 2: daytime morning-skate starting goalies, line combinations, and injury statuses), refreshes pre-game rolling features, and outputs predictions in one command.

```bash
# First-Time Setup (or periodic monthly retrain):
# 1. Backfill historical seasons + current season to date
sports-ml ingest --sport hockey --seasons 20222023,20232024,20242025,20252026 --current-season

# 2. Auto-build features & fit Stage 1 (Win Prob) + Stage 2 (Player Stats) models
sports-ml train --sport hockey --include-stage2

# ------------------------------------------------------------------------------
# Day-to-Day Usage (1 Command — connects live by default for stats, injuries & starters):
# Sync current season to date, pull live rosters/goalies/injuries, and predict tonight's slate
sports-ml predict --sport hockey --date 2026-10-25

# Predict a specific matchup with Stage 2 Optional Player Stat Projections enabled
sports-ml predict --sport hockey --date 2026-10-25 --matchup TOR@BOS --include-players
```

#### B. Granular Model Development & Evaluation Commands
Used when experimenting with new features, auditing data, or measuring out-of-sample model accuracy without re-running the entire pipeline:

| Command | Purpose | Required for Daily Predictions? |
| :--- | :--- | :--- |
| `sports-ml ingest` | Downloads/caches raw NHL API & MoneyPuck data into Silver Parquet tables. | Automatic during `predict` |
| `sports-ml features build` | Recomputes pre-game rolling windows, Elo, goalie GSAx, and roster weights into Gold Parquet tables without hitting the network. | Automatic during `train` and `predict` |
| `sports-ml backtest` | **"Time Machine" Evaluator**: Simulates training and predicting across historical seasons month-by-month to score Log Loss, Brier Score, and calibration against baselines. | **No** (Evaluation / R&D only) |
| `sports-ml train` | Fits the production LightGBM + Calibrator bundle on all completed games and saves to `artifacts/` with a `metadata.json` schema fingerprint. | Run once initially, then periodically (e.g., weekly/monthly) |
| `sports-ml score-predictions` | Joins the append-only live prediction ledger (`data/predictions/`) against completed game/player boxscores to score real-world pre-game predictions. | **No** (Live season audit only) |

```bash
# Sync only the current season to date (or a specific date window)
sports-ml ingest --sport hockey --current-season
sports-ml ingest --sport hockey --start-date 2026-10-17 --end-date 2026-10-25

# Rebuild Gold feature tables locally after tweaking feature parameters (zero network calls)
sports-ml features build --sport hockey

# Run walk-forward time-series backtesting to evaluate model accuracy (R&D / validation only)
sports-ml backtest --sport hockey --stage 1
sports-ml backtest --sport hockey --stage all
```

### Python Package API
```python
from datetime import date
from sports_ml.sports.hockey.models.predictor import HockeyMatchupPredictor

predictor = HockeyMatchupPredictor.from_artifacts("artifacts/hockey/latest")

# Stage 1 only (Mandatory Calibrated Win Probability)
game_pred = predictor.predict_matchup(
    home_team="BOS",
    away_team="TOR",
    game_date=date(2026, 10, 25),
    include_players=False,
)
# -> game_pred.home_win_prob == 0.564, game_pred.away_win_prob == 0.436, game_pred.player_projections == None

# Stage 1 + Stage 2 (With Player Stat Projections for both teams)
full_pred = predictor.predict_matchup(
    home_team="BOS",
    away_team="TOR",
    game_date=date(2026, 10, 25),
    include_players=True,
)
# -> full_pred.home_win_prob == 0.564
# -> full_pred.player_projections.home_skaters (Polars DataFrame of E[G], E[A], E[PTS], E[SOG], E[TOI], P(1+ PT))
# -> full_pred.player_projections.away_skaters
# -> full_pred.player_projections.goalies
```

---

## 7. Technology Stack Summary

| Layer | Package | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Modern typing (`Self`, `Protocol`, union types) and performance. |
| **DataFrame & Storage** | `polars` + `pyarrow` | Expressive lazy window functions, zero-copy Parquet I/O, strict schema enforcement, no index-alignment bugs. |
| **Data Contracts** | `pydantic >= 2.0` | Validated configuration, API payload parsing, and prediction output schemas. |
| **HTTP & Ingestion** | `httpx` + `tenacity` | Async/sync HTTP client with deterministic disk caching and exponential backoff retries. |
| **ML & Calibration** | `lightgbm`, `scikit-learn`, `scipy` | State-of-the-art GBDTs for tabular classification/Poisson count regression, plus isotonic/Platt calibration. |
| **CLI & Formatting** | `typer`, `rich` | Clean CLI subcommands and readable matchup/player projection tables. |
| **Linting, Typing & Testing** | `ruff`, `mypy`, `pytest`, `pytest-cov` | Fast linting/formatting, strict static type checking, and offline unit/anti-leakage test coverage. |

---

## 8. Coding Standards, Unit Testing Strategy & GitHub CI

### 8.1 Coding Standards, Linting & Type Checking
All code quality rules are configured centrally in `pyproject.toml` and enforced locally and in CI:
1. **`ruff` (Linting + Formatting)**:
   * Enforces `ruff check .` and `ruff format --check .`.
   * Enabled rule sets: `E` / `F` (pycodestyle & Pyflakes), `I` (import sorting), `B` (flake8-bugbear), `UP` (pyupgrade for Python 3.11+ syntax), `SIM` (code simplification), and `NPY` (NumPy/array best practices).
2. **`mypy` (Static Type Checking)**:
   * Enforces explicit type annotations on all public functions, methods, and `SportAdapter` protocol implementations across `src/sports_ml/` and `tests/`.
3. **Polars + Pydantic Data Schema Validation**:
   * Silver and Gold Parquet tables are validated against canonical Polars column/dtype contracts (`Int64`, `Float64`, `Datetime("us", "UTC")`) and non-null constraints to prevent silent `NaN` propagation.

### 8.2 Offline Unit Testing Strategy (`pytest`)
All unit tests in `tests/` run **100% offline** in seconds using deterministic synthetic Polars DataFrames and saved JSON/CSV fixtures (never querying live external APIs during tests). Testing is focused on four high-leverage pillars:

1. **"Time Machine" Anti-Leakage Invariance Tests (`test_temporal_leakage.py`, `test_hockey_features.py`)**:
   * Generates features for a synthetic multi-game timeline, records the feature vector at event $t$, then mutates scores/stats for all events $\ge t$ and asserts that the feature vector at event $t$ remains **bit-for-bit identical**.
2. **Mathematical & Probability Invariants (`test_ratings.py`, `test_stage1_model.py`, `test_stage2_player.py`)**:
   * Pairwise Elo updates are strictly zero-sum ($\Delta E_{\text{home}} + \Delta E_{\text{away}} = 0$), and off-season regression moves ratings toward `1500`.
   * Stage 1 calibrated probabilities satisfy $0 < P < 1$ and $\sum P = 1.0$.
   * Stage 2 player projections (TOI, goals, assists, shots) are strictly non-negative ($\ge 0$) and respond monotonically to Stage 1 team scoring environment shifts.
3. **Temporal & Cold-Start Edge Cases (`test_schedule.py`, `test_hockey_features.py`)**:
   * Opening Night games and rookie/goalie debuts with `0` prior appearances resolve cleanly to shrunk priors with zero `NaN`s.
   * Schedule calculations handle month/year boundaries (e.g., Dec 31 $\rightarrow$ Jan 1), overtime/shootouts, and same-day doubleheaders (`event_seq = 1` vs. `2`).
4. **Ingestion Parser & Fallback Tests (`test_ingestion.py`)**:
   * Validates NHL API JSON and MoneyPuck CSV reconciliation, historical team tricode normalization (`ARI` $\rightarrow$ `UTA`, `S.J` $\rightarrow$ `SJS`), and graceful fallback when MoneyPuck data lags behind NHL API boxscores.

### 8.3 GitHub Actions Continuous Integration (`.github/workflows/ci.yml`)
A single, fast, least-privilege CI workflow runs on every `push` and `pull_request` to `main`:
* **Permissions**: `contents: read` (least privilege).
* **Environment**: Ubuntu runner, Python 3.11+ with `uv` / lockfile (`uv.lock`) dependency caching.
* **Verification Steps (< 60 seconds, 100% offline)**:
  1. **Git Hygiene / Large-File Check**: Fails if any files under `data/`, `artifacts/`, or binary `.parquet`/`.pkl` files are tracked in git.
  2. `ruff check .` (linting)
  3. `ruff format --check .` (formatting check)
  4. `mypy src tests` (static type checking)
  5. `pytest --cov=sports_ml` (unit, schema, and anti-leakage test suite)

---

## 9. Operational Guardrails, Reproducibility & Auditability

### 9.1 Model Artifact Fingerprinting (`metadata.json` Checksums)
To prevent silent feature-schema drift between training and daily inference, `core/registry.py` writes a `metadata.json` alongside every saved model bundle (`artifacts/<sport>/<version>/metadata.json`) containing:
* Exact ordered list of Stage 1 and Stage 2 feature column names, Polars dtypes, and a SHA-256 schema fingerprint.
* Training data coverage (`min_event_date`, `max_event_date`, total events) and git commit SHA.
* Walk-forward validation metrics (`log_loss`, `brier_score`, `ece`) recorded at training time.
* Before `predict` executes, `registry.py` validates that the live input DataFrame matches the saved schema fingerprint and raises an actionable error if retraining is required.

### 9.2 Append-Only Live Prediction Ledger (`data/predictions/`)
Every invocation of `sports-ml predict` automatically appends the generated pre-game predictions (`predicted_at_utc`, `event_id`, `model_version`, `home_win_prob`, `away_win_prob`, confidence flags, and optional Stage 2 player projections) to `data/predictions/<sport>_predictions.parquet`.
* Running `sports-ml score-predictions --sport hockey` joins this ledger against subsequently completed `FINAL` games in `data/silver/` to evaluate real-world, live pre-game accuracy over the course of a season.

### 9.3 Pre-Prediction Data Confidence & Lineup Warnings
Live pre-game predictions surface explicit confidence indicators in both the `Predictor` return object and the Rich CLI table:
* **Starting Goalie Status**: Distinguishes `CONFIRMED` (official pre-game starter) from `PROJECTED` (inferred via primary non-B2B starter heuristic).
* **Data Freshness**: Displays the latest completed game date present in `data/silver/` and whether MoneyPuck $xG$ data is current or using the NHL API boxscore fallback.
* **Prior Shrinkage Flags**: Flags debuting rookies or early-season teams where Bayesian prior shrinkage accounts for $>50\%$ of the feature weight.

### 9.4 Deterministic Reproducibility & Git Hygiene
* **Lockfile (`uv.lock`)**: Guarantees identical package versions across local development and GitHub Actions CI.
* **Central Seed (`RANDOM_SEED = 42` in `config.py`)**: Propagated to all GBDT estimators, cross-validation splitters, and calibrators for deterministic training.
* **Strict `.gitignore`**: Excludes `data/`, `artifacts/`, `.venv/`, and tool caches so raw data and model binaries are never committed to the repository.


