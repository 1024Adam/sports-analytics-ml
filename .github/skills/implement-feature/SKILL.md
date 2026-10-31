---
name: implement-feature
description: "Use when: implementing a new feature, enhancement, CLI option, model improvement, or pipeline capability directly from the user's prompt description, writing offline unit and anti-leakage tests, running ruff/mypy/pytest verification gates, and updating relevant documentation."
argument-hint: "Describe the feature, enhancement, or capability to implement and verify..."
user-invocable: true
disable-model-invocation: false
---

# Feature Implementation Workflow

## When to Use
- When the user invokes `/implement-feature <description>` or asks to build a standalone feature, enhancement, or capability described directly in their prompt (without requiring a numbered milestone in `docs/roadmap.md`).

---

## Step-by-Step Execution Procedure

### Step 1: Parse Prompt Requirements & Load Architecture Context
1. Analyze the feature description provided in the user's prompt to determine:
   - Target scope (`core/` sport-agnostic engine vs. `sports/hockey/`, `sports/soccer/`, or CLI/tooling).
   - Affected pipeline layer(s) (ingestion, Silver reconciliation, Gold feature engineering, Stage 1/Stage 2 models, evaluation/backtesting, or CLI).
2. Read the corresponding sections of `./docs/architecture.md`, `./docs/hockey-data-and-features.md`, or `./docs/soccer-data-and-features.md` and inspect existing modules to confirm schemas, contracts, and integration points.
3. Register the feature's implementation deliverables and test tasks in `#tool:todo`.

### Step 2: Implement Deliverables Incrementally
1. Respect the strict boundary between `src/sports_ml/core/` (sport-agnostic primitives supporting NHL, EPL, MLB, and PGA across `head_to_head` and `multi_entrant_field` topologies) and `src/sports_ml/sports/<sport>/` (sport-specific logic).
2. Enforce zero temporal data leakage: any pre-game feature, rolling window, or rating update must use strictly prior completed events (`shift(1)` / `< start_time_utc` ordered by `["start_time_utc", "event_seq", "event_id"]`).
3. Add complete Python 3.11+ type annotations to all new or modified functions, methods, and classes.
4. Ensure all stochastic operations reference `RANDOM_SEED = 42` from `src/sports_ml/config.py`.
5. Never commit or generate untracked data files outside git-ignored paths or temporary `pytest` `tmp_path` fixtures.

### Step 3: Write Offline Unit & Anti-Leakage Tests
1. Create or update the relevant `pytest` test modules in `tests/unit/` or `tests/integration/` covering normal behavior and edge cases.
2. All tests MUST run 100% offline using deterministic synthetic Polars DataFrames or local JSON/CSV fixtures.
3. If the feature adds or modifies any rating, rolling window, shrinkage prior, or feature module, include the **Future-Mutation Invariance Test** (see `.github/skills/leak-free-feature-audit/SKILL.md`).

### Step 4: Run the Mandatory Quality Gate
Run all four verification commands and fix any errors before proceeding:
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=sports_ml
```

### Step 5: Update Documentation & Summarize
1. Once all four checks in Step 4 exit with code `0`, update any directly affected documentation (such as `./README.md`, `./docs/architecture.md`, `./docs/hockey-data-and-features.md`, or `./docs/soccer-data-and-features.md`) if CLI options, schemas, or feature definitions changed.
2. Report a concise summary of created/updated files, how the prompt's feature requirements were satisfied, and test/verification results.
