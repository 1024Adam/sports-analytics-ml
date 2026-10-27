# Sports Analytics ML (`sports-analytics-ml`)

A modular, leak-free, multi-sport machine learning platform built in **Python 3.11+** using **Polars**, **PyArrow**, **Pydantic**, **LightGBM**, **scikit-learn**, and **Typer/Rich**.

Supports **NHL Hockey (`--sport hockey`)** and **English Premier League (EPL) Soccer (`--sport soccer`)**:
- **NHL Hockey**: Ingests schedule, roster, boxscore, and advanced shot-quality ($xG$, Corsi, Fenwick, Goalie GSAx) data from the **Official NHL Web API** (`https://api-web.nhle.com/v1/`) and **MoneyPuck**, engineers strictly pre-game temporal features, and executes a **Two-Stage Prediction Pipeline**:
  1. **Stage 1 — Mandatory Calibrated Game Outcome Model**: Produces calibrated win probabilities $P(\text{Home Win})$ and $P(\text{Away Win}) = 1 - P(\text{Home Win})$, alongside expected team scoring pace $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$.
  2. **Stage 2 — Optional Player Matchup Stat Projections**: Conditioned on Stage 1's game environment and opponent strength, predicts skater usage ($\widehat{\text{TOI}}_{\text{ev}}, \widehat{\text{TOI}}_{\text{pp}}$), counting stats ($E[\text{G}], E[\text{A}], E[\text{PTS}], E[\text{SOG}]$), milestone probabilities ($P(\text{G} \ge 1), P(\text{PTS} \ge 1)$), and goalie metrics ($E[\text{SA}], E[\text{SV}], E[\text{GA}], E[\text{SV\%}]$) reconciled via damped proportional scaling ($\gamma = 0.50$).
- **EPL Soccer**: Ingests fixtures, squad availability/set-piece order, and match stats from the **Official FPL / Premier League API**, **Understat** ($npxG$, $xG$, $xA$, $xGChain$, $PPDA$, deep completions), and **Football-Data.co.uk**, and executes:
  1. **Stage 1 — Mandatory Calibrated 3-Way (`1X2`) & Goals Environment Model**: Combines 3-way Elo (with promotion/relegation priors), Dixon-Coles bivariate Poisson low-score correlation ($\rho$), and Multiclass LightGBM to produce calibrated probabilities $P(\text{Home Win}) + P(\text{Draw}) + P(\text{Away Win}) = 1.0$ and expected goals $(\hat{\lambda}_{\text{home}}, \hat{\lambda}_{\text{away}})$, evaluated via **Ranked Probability Score (RPS)**, 3-class Log Loss, and Brier Score.
  2. **Stage 2 — Optional Outfield & Goalkeeper Projections**: Projects outfield player minutes ($\in [0, 90]$), goals, assists, goal involvements ($G+A$), shots, SOT, milestone probabilities, and goalkeeper saves, goals against, save %, and clean-sheet probability $P(\text{CS})$.

---

## Documentation

- [System Architecture Blueprint](./docs/architecture.md)
- [NHL Data Contracts & Feature Specification](./docs/data-and-features.md)
- [EPL Soccer Data Contracts & Feature Specification](./docs/soccer-data-and-features.md)
- [Phased Implementation Roadmap](./docs/roadmap.md)
- [Contributing & Anti-Leakage Guidelines](./CONTRIBUTING.md)
- [Security Policy](./SECURITY.md)
- [MIT License](./LICENSE)

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
# 1. Backfill all seasons from a starting year through the current season to date
uv run sports-ml ingest --sport hockey --from-year 2023

# 2. Auto-build Gold features & fit Stage 1 (Win Prob) + Stage 2 (Player Stats) models
uv run sports-ml train --sport hockey --include-stage2

# 3. Predict tonight's slate (connects live by default; use --offline for local cache only)
uv run sports-ml predict --sport hockey --date 2026-10-25

# 4. Predict a specific matchup with Stage 2 Player Stat Projections (broken down by team & F/D/G)
uv run sports-ml predict --sport hockey --date 2026-10-25 --matchup TOR@BOS --include-players

# 5. Predict a team's next scheduled game, or their next N scheduled games
uv run sports-ml predict --sport hockey --team COL
uv run sports-ml predict --sport hockey --team COL --games 5

# --- EPL Soccer (--sport soccer) ---
# 1. Backfill all EPL seasons from a starting year through the active season (FPL + Understat + Football-Data)
uv run sports-ml ingest --sport soccer --from-year 2023

# 2. Train Stage 1 (3-Way 1X2 + Goals) and Stage 2 (Outfield & GK Projections)
uv run sports-ml train --sport soccer --include-stage2

# 3. Predict an EPL matchweek or specific matchup with Stage 2 Outfield & GK Projections
uv run sports-ml predict --sport soccer --date 2026-10-25 --matchup MCI@ARS --include-players
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

# Compact and deduplicate existing local raw caches in place (.csv.gz / .json.gz)
uv run sports-ml compact-cache
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

# Predict a team's next N scheduled games in chronological order
next_games = predictor.predict_team_next_games("COL", n_games=3, include_players=False)
```

---

## Copilot Slash-Command Skills (`.github/skills/`)

### Run-Action Skills
- `/predict-matchup` ([SKILL.md](./.github/skills/predict-matchup/SKILL.md)): Run `sports-ml predict` for a slate date, a specific `AWAY@HOME` matchup, or a team's next $N$ games (`--team` / `--games`), optionally with `--include-players` (Stage 2) or `--offline`.
- `/ingest-data` ([SKILL.md](./.github/skills/ingest-data/SKILL.md)): Run `sports-ml ingest` for `--from-year` (all seasons from start year to current), `--current-season`, `--start-date`/`--end-date`, `--seasons`, or `--offline` Silver Parquet reconciliation.
- `/train-models` ([SKILL.md](./.github/skills/train-models/SKILL.md)): Run `sports-ml features build` and `sports-ml train` (`--include-stage2`) and inspect `metadata.json` SHA-256 schema fingerprints.
- `/evaluate-and-score` ([SKILL.md](./.github/skills/evaluate-and-score/SKILL.md)): Run walk-forward `sports-ml backtest` and score the live `data/predictions/` audit ledger (`sports-ml score-predictions`).

### Engineering & Architecture Skills
- `/implement-feature` ([SKILL.md](./.github/skills/implement-feature/SKILL.md)): Implement and verify a standalone feature or enhancement directly from a prompt description.
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
