# Sports Analytics ML (`sports-analytics-ml`)

A modular, leak-free, multi-sport machine learning platform built in **Python 3.11+** using **Polars**, **PyArrow**, **Pydantic**, **LightGBM**, **scikit-learn**, and **Typer/Rich**.

Starting with **NHL Hockey**, the platform ingests schedule, roster, boxscore, and advanced shot-quality ($xG$, Corsi, Fenwick, Goalie GSAx) data from the **Official NHL Web API** (`https://api-web.nhle.com/v1/`) and **MoneyPuck**, engineers strictly pre-game temporal features, and executes a **Two-Stage Prediction Pipeline**:

1. **Stage 1 — Mandatory Calibrated Game Outcome Model**: Produces calibrated win probabilities $P(\text{Home Win})$ and $P(\text{Away Win}) = 1 - P(\text{Home Win})$, alongside expected team scoring pace $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$.
2. **Stage 2 — Optional Player Matchup Stat Projections**: Conditioned on Stage 1's game environment and opponent strength, predicts skater usage ($\widehat{\text{TOI}}_{\text{ev}}, \widehat{\text{TOI}}_{\text{pp}}$), counting stats ($E[\text{G}], E[\text{A}], E[\text{PTS}], E[\text{SOG}]$), milestone probabilities ($P(\text{G} \ge 1), P(\text{PTS} \ge 1)$), and goalie metrics ($E[\text{SA}], E[\text{SV}], E[\text{GA}], E[\text{SV\%}]$) reconciled via damped proportional scaling ($\gamma = 0.50$).

---

## Documentation

- [System Architecture Blueprint](./docs/architecture.md)
- [Data Contracts & Feature Specification](./docs/data-and-features.md)
- [Phased Implementation Roadmap](./docs/roadmap.md)

---

## Quickstart

### 1. Install Dependencies
```bash
uv sync --dev
```

### 2. Streamlined Daily & Setup CLI Workflow (`sports-ml`)
```bash
# 1. Backfill historical seasons + current season to date
uv run sports-ml ingest --sport hockey --seasons 20232024,20242025,20252026 --current-season

# 2. Auto-build Gold features & fit Stage 1 (Win Prob) + Stage 2 (Player Stats) models
uv run sports-ml train --sport hockey --include-stage2

# 3. Predict tonight's slate (connects live by default; use --offline for local cache only)
uv run sports-ml predict --sport hockey --date 2026-10-25

# 4. Predict a specific matchup with Stage 2 Player Stat Projections enabled
uv run sports-ml predict --sport hockey --date 2026-10-25 --matchup TOR@BOS --include-players
```

### 3. Developer & Audit Commands
```bash
# Recompute Gold pre-game feature matrices locally without network calls
uv run sports-ml features build --sport hockey

# Run walk-forward chronological backtesting against baselines
uv run sports-ml backtest --sport hockey --stage all

# Score the append-only live prediction ledger (data/predictions/) against completed games
uv run sports-ml score-predictions --sport hockey
```

### 4. Python Package API
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

# Stage 1 + Stage 2 (With Skater & Goalie Projections)
full_pred = predictor.predict_matchup(
    home_team="BOS",
    away_team="TOR",
    game_date=date(2026, 10, 25),
    include_players=True,
)
```

---

## Quality Gate & Anti-Leakage Verification
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=sports_ml
```
