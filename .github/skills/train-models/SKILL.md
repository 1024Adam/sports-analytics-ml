---
name: train-models
description: "Use when: building pre-game Gold feature tables (`sports-ml features build`) or training production Stage 1 and Stage 2 model bundles (`sports-ml train`) for `--sport hockey` or `--sport soccer`, updating model artifacts in `artifacts/<sport>/`, or refreshing SHA-256 schema fingerprints (`metadata.json`)."
argument-hint: "Optional --sport (hockey|soccer), --include-stage2, --version <tag>, or features-only..."
user-invocable: true
disable-model-invocation: false
---

# Train Models & Build Features Workflow (`/train-models`)

## When to Use
- When the user invokes `/train-models` or asks to build Gold features, fit Stage 1 (Calibrated Win Probability) and/or Stage 2 (Player Matchup Stat Projection) models, or resolve a `SchemaMismatchError` after feature changes.

---

## Step-by-Step Execution Procedure

### Step 1: Verify Silver Data Availability
Check that `data/silver/hockey/games.parquet` and `data/silver/hockey/team_game_logs.parquet` exist and contain completed (`FINAL`) games:
- If Silver tables are missing, run `/ingest-data` (`uv run sports-ml ingest --sport hockey --current-season`) first.

### Step 2: Rebuild Pre-Game Gold Features (Optional Standalone or Auto-Chained)
- If the user only wants to recompute the leak-free Gold feature tables (`data/gold/hockey/stage1_game_features.parquet` and `stage2_player_features.parquet`) without training models, run:
  ```bash
  uv run sports-ml features build --sport hockey
  ```
- Otherwise, `sports-ml train` automatically builds/updates the Gold feature tables before fitting the models.

### Step 3: Run `sports-ml train`
Choose whether to include Stage 2 player projection models (`--include-stage2` is recommended so both Stage 1 and Stage 2 are ready for inference):

```bash
# Train both Stage 1 and Stage 2 models (Hockey or Soccer)
uv run sports-ml train --sport hockey --include-stage2 --version v1
uv run sports-ml train --sport soccer --include-stage2 --version v1

# Train Stage 1 only
uv run sports-ml train --sport hockey --no-include-stage2 --version v1
uv run sports-ml train --sport soccer --no-include-stage2 --version v1
```

### Step 4: Inspect Saved Bundle & `metadata.json` Fingerprint
Read `artifacts/hockey/latest/metadata.json` and summarize for the user:
1. **Model Version & Git Commit SHA**: `model_version`, `git_commit_sha`.
2. **Training Coverage**: `min_event_date`, `max_event_date`, and `total_events`.
3. **Schema Fingerprints**: `schema_sha256` (Stage 1) and `stage2_schema_sha256` (Stage 2, if enabled).
4. **Validation Metrics**: `log_loss`, `brier_score`, `ece`, and `accuracy`.
