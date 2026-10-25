---
name: implement-milestone
description: "Use when: executing a numbered milestone from docs/roadmap.md (e.g., Milestone 1 through Milestone 6), building its deliverables, running ruff/mypy/pytest verification gates, and updating milestone completion status."
argument-hint: "Milestone number (1-6) from docs/roadmap.md to implement and verify..."
user-invocable: true
disable-model-invocation: false
---

# Milestone Implementation Workflow

## When to Use
- When the user invokes `/implement-milestone <N>` or asks to build the next milestone in `docs/roadmap.md`.

---

## Step-by-Step Execution Procedure

### Step 1: Load Blueprint & Target Milestone Context
1. Read `./docs/roadmap.md` to identify the target milestone (either the number specified by the user, or the first milestone marked `[ ] Not Started` / `[~] In Progress`).
2. Read the corresponding sections of `./docs/architecture.md` and `./docs/data-and-features.md` to confirm exact file paths, function signatures, Polars schemas, and mathematical formulas.
3. Register the milestone's deliverables and test tasks in `#tool:todo`.

### Step 2: Implement Deliverables Incrementally
1. Respect the strict boundary between `src/sports_ml/core/` (sport-agnostic primitives supporting NHL, MLB, and PGA) and `src/sports_ml/sports/hockey/` (NHL-specific logic).
2. Add complete Python 3.11+ type annotations to all functions, methods, and classes.
3. Ensure all random seeds reference `RANDOM_SEED = 42` from `src/sports_ml/config.py`.
4. Never commit or generate untracked data files outside git-ignored paths or temporary `pytest` `tmp_path` fixtures.

### Step 3: Write Offline Unit & Anti-Leakage Tests
1. Create or update the `pytest` test modules listed under the milestone's **Verification & Acceptance Criteria** in `./docs/roadmap.md`.
2. All tests MUST run 100% offline using deterministic synthetic Polars DataFrames or local JSON/CSV fixtures.
3. For any rating, rolling, or feature module, include the **Future-Mutation Invariance Test** (see `.github/skills/leak-free-feature-audit/SKILL.md`).

### Step 4: Run the Mandatory Quality Gate
Run all four verification commands and fix any errors before proceeding:
```bash
ruff check .
ruff format --check .
mypy src tests
pytest --cov=sports_ml
```

### Step 5: Update `docs/roadmap.md` & Summarize
1. Once all four checks in Step 4 exit with code `0`, edit `./docs/roadmap.md` to:
   - Mark the milestone's **Status** as `` `[x] Completed` ``.
   - Check off `- [x]` every completed deliverable and verification item in that milestone.
2. Report a concise summary of created/updated files, test coverage results, and which milestone is next.
