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
# Optional: install CLI globally in editable mode so you can run `sports-ml` without `uv run`
uv tool install --editable .
```

### 2. Streamlined Daily & Setup CLI Workflow (`sports-ml`)
```bash
# 1. Backfill historical seasons + current season to date
uv run sports-ml ingest --sport hockey --seasons 20232024,20242025,20252026 --current-season

# 2. Auto-build Gold features & fit Stage 1 (Win Prob) + Stage 2 (Player Stats) models
uv run sports-ml train --sport hockey --include-stage2

# 3. Predict tonight's slate (connects live by default; use --offline for local cache only)
uv run sports-ml predict --sport hockey --date 2026-10-25

# 4. Predict a specific matchup with Stage 2 Player Stat Projections (broken down by team & F/D/G)
uv run sports-ml predict --sport hockey --date 2026-10-25 --matchup TOR@BOS --include-players
```

### 3. Developer & Audit Commands
```bash
# Verify live connectivity and schema contracts across NHL API, MoneyPuck & DailyFaceoff
uv run sports-ml check-sources --sport hockey

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

## Copilot Slash-Command Skills (`.github/skills/`)

### Run-Action Skills
- `/predict-matchup` ([SKILL.md](./.github/skills/predict-matchup/SKILL.md)): Run `sports-ml predict` for a slate date or specific `AWAY@HOME` matchup, optionally with `--include-players` (Stage 2) or `--offline`.
- `/ingest-data` ([SKILL.md](./.github/skills/ingest-data/SKILL.md)): Run `sports-ml ingest` for `--current-season`, `--start-date`/`--end-date`, `--seasons`, or `--offline` Silver Parquet reconciliation.
- `/train-models` ([SKILL.md](./.github/skills/train-models/SKILL.md)): Run `sports-ml features build` and `sports-ml train` (`--include-stage2`) and inspect `metadata.json` SHA-256 schema fingerprints.
- `/evaluate-and-score` ([SKILL.md](./.github/skills/evaluate-and-score/SKILL.md)): Run walk-forward `sports-ml backtest` and score the live `data/predictions/` audit ledger (`sports-ml score-predictions`).

### Engineering & Architecture Skills
- `/implement-milestone` ([SKILL.md](./.github/skills/implement-milestone/SKILL.md)): Execute and verify roadmap milestones.
- `/leak-free-feature-audit` ([SKILL.md](./.github/skills/leak-free-feature-audit/SKILL.md)): Audit temporal window functions and run future-mutation invariance tests.
- `/nhl-data-integration` ([SKILL.md](./.github/skills/nhl-data-integration/SKILL.md)): Reference for Official NHL Web API v1, MoneyPuck CSV/ZIPs, and 2-Tier pre-game lineups.
- `/onboard-sport-plugin` ([SKILL.md](./.github/skills/onboard-sport-plugin/SKILL.md)): Playbook for adding MLB Baseball or PGA Golf plugins.

---

## Quality Gate & Anti-Leakage Verification
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=sports_ml
```
