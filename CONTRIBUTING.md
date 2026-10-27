# Contributing to Sports Analytics ML (`sports-analytics-ml`)

Thank you for contributing to `sports-analytics-ml`! Because this platform models chronological sports outcomes and player projections, all contributions must adhere to strict architectural, anti-leakage, and verification invariants.

---

## 1. Core Architectural & Anti-Leakage Invariants

Every feature, rating system, or model update **must** guarantee zero future data leakage:

1. **Strict Left-Closed Temporal Windows (`.shift(1)`)**
   - For any event occurring at `(start_time_utc, event_seq, event_id)` at timestamp $T$, only completed events with timestamp strictly $< T$ may contribute to pre-game features.
   - Always use the core rolling/exposure utilities in [`src/sports_ml/core/temporal.py`](./src/sports_ml/core/temporal.py) rather than ad-hoc rolling expressions that risk including the current game row.
2. **Walk-Forward Chronological Validation Only**
   - **Never** use random `train_test_split` or standardshuffled $K$-fold cross-validation on time-series sports data.
   - All model training, hyperparameter tuning, and probability calibration (Platt scaling, isotonic regression, temperature scaling) must fit strictly on historical folds via [`src/sports_ml/core/validation.py`](./src/sports_ml/core/validation.py).
3. **Cold-Start & Season-Boundary Priors**
   - New features must explicitly handle games $1\text{--}10$ of a season (or newly promoted clubs in soccer) using Bayesian shrinkage toward prior-season baselines or league-average priors.
4. **Realistic Pre-Game Information Boundary**
   - Never use post-game box-score actuals (e.g., actual Time on Ice, actual starting goalie after puck drop, or post-match injury status) as pre-game model inputs.
5. **Core vs. Sport-Plugin Separation**
   - Sport-agnostic engines (`contracts.py`, `temporal.py`, `ratings.py`, `schedule.py`, `validation.py`, `calibration.py`, `evaluation.py`) belong in [`src/sports_ml/core/`](./src/sports_ml/core/).
   - Sport-specific ingestion, schemas, features, and two-stage predictors belong in [`src/sports_ml/sports/<sport>/`](./src/sports_ml/sports/).

---

## 2. Development Setup

Prerequisites: **Python 3.11+** and [**`uv`**](https://docs.astral.sh/uv/).

```bash
# Clone the repository and sync all development dependencies
uv sync --dev
```

---

## 3. Testing & Quality Gates

All pull requests are gated by `.github/workflows/ci.yml`. Before opening a PR, run the full local verification suite:

```bash
# 1. Linting & import sorting
uv run ruff check .

# 2. Code formatting check
uv run ruff format --check .

# 3. Strict static type checking
uv run mypy src tests

# 4. Offline deterministic test suite
uv run pytest -q
```

### Test Suite Rules
- **100% Offline & Deterministic**: Unit and integration tests in `tests/` must **never** make live network calls to external APIs (NHL Web API, MoneyPuck, FPL, Understat, etc.). Mock HTTP transports (`httpx.MockTransport`) or use local fixtures.
- **Future-Mutation Invariance**: When adding a new rolling or rating feature, add or extend an anti-leakage test verifying that mutating box-score statistics of games at $t \ge T$ produces **bit-identical** pre-game feature values at timestamp $T$.

---

## 4. Pull Request Checklist

When submitting a pull request, verify:
- [ ] Temporal boundaries (`.shift(1)` / walk-forward splits) are preserved with zero future leakage.
- [ ] `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src tests`, and `uv run pytest` pass cleanly.
- [ ] No raw cached datasets (`data/`), local model binaries (`artifacts/`), or secrets are staged for commit.
- [ ] Relevant design documentation in [`docs/`](./docs/) and [`README.md`](./README.md) is updated if schemas, features, or CLI flags changed.
