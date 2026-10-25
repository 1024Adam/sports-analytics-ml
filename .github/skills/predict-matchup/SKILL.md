---
name: predict-matchup
description: "Use when: running pre-game matchup predictions (`sports-ml predict`), generating Stage 1 calibrated win probabilities, projecting Stage 2 skater and goalie stats (`--include-players`), predicting tonight's NHL slate or a specific matchup (`--matchup AWAY@HOME`), or interpreting pre-prediction confidence warnings."
argument-hint: "Optional date (YYYY-MM-DD), matchup (e.g., TOR@BOS), --include-players, or --offline..."
user-invocable: true
disable-model-invocation: false
---

# Predict Matchup Workflow (`/predict-matchup`)

## When to Use
- When the user invokes `/predict-matchup` or asks to predict tonight's NHL slate or a specific head-to-head matchup.
- When running **Stage 1 only** (Mandatory Calibrated Game Win Probabilities & Expected Team Goals) or **Stage 1 + Stage 2** (`--include-players` for active roster skater & goalie stat projections).

---

## Step-by-Step Execution Procedure

### Step 1: Parse Target Date, Matchup Filter & Stage Mode
Determine the user's desired prediction parameters:
1. **Sport (`--sport`)**: Defaults to `hockey`.
2. **Date (`--date YYYY-MM-DD`)**: Defaults to today's UTC/local slate date if omitted.
3. **Matchup Filter (`--matchup AWAY@HOME`)**: Optional (e.g., `TOR@BOS` or `BOS vs TOR`). If omitted, predicts all scheduled games on the target date.
4. **Stage 2 Player Projections (`--include-players` / `--no-include-players`)**:
   - Omit (or pass `--no-include-players`) when only Stage 1 game win probability is needed.
   - Pass `--include-players` when skater (`E[TOI]`, `E[G]`, `E[A]`, `E[PTS]`, `E[SOG]`, `P(1+ PT)`) and goalie (`E[SA]`, `E[SV]`, `E[GA]`, `E[SV%]`) projections are requested.
5. **Network Mode (`--offline`)**:
   - By default, `sports-ml predict` connects live to sync current-season completed games and schedules before predicting.
   - Pass `--offline` only if the user explicitly wants to predict strictly from already-cached local Silver Parquet tables.

### Step 2: Verify Model Bundle Availability
Check that a trained model bundle exists at `artifacts/<sport>/latest/metadata.json`:
- If `artifacts/hockey/latest/metadata.json` is missing (or if `--include-players` is requested but `stage2_model.pkl` is absent), run:
  ```bash
  uv run sports-ml train --sport hockey --include-stage2
  ```

### Step 3: Execute `sports-ml predict`
Run the appropriate CLI command via bash:

```bash
# Predict full slate for a date (Stage 1 Win Probabilities only)
uv run sports-ml predict --sport hockey --date 2026-10-25

# Predict a specific matchup with Stage 2 Player Stat Projections
uv run sports-ml predict --sport hockey --date 2026-10-25 --matchup TOR@BOS --include-players

# Predict offline using local cached Silver tables only
uv run sports-ml predict --sport hockey --date 2026-10-25 --matchup TOR@BOS --include-players --offline
```

### Step 4: Interpret Output & Confidence Warnings
Present the results clearly to the user:
1. **Stage 1 Summary**: Report `P(Home Win)`, `P(Away Win)`, and expected regulation/OT scoring environment (`E[Home Goals]`, `E[Away Goals]`).
2. **Confidence & Lineup Warnings**: Highlight any surfaced warnings:
   - **Starting Goalie Status**: Whether each starter is `CONFIRMED` (`CONFIRMED_NHL_OFFICIAL` / `CONFIRMED_MORNING_SKATE`) or `PROJECTED_HEURISTIC`.
   - **Data Freshness**: Whether MoneyPuck $xG$ data is current or using the NHL API boxscore fallback (`has_moneypuck_xg = False`).
   - **Prior Shrinkage Flags**: Early-season teams ($n < 8$ games) or low-sample goalies ($n < 15$ starts) where Bayesian prior shrinkage carries $>50\%$ weight.
3. **Stage 2 Player Projections (if `--include-players`)**: Summarize top projected skaters for both teams (`E[PTS]`, `E[G]`, `E[A]`, `E[SOG]`, `E[TOI]`, `P(1+ PT)`) and starting goalie projections.
4. **Audit Ledger Note**: Confirm that the prediction was appended to `data/predictions/hockey_predictions.parquet` for future scoring via `/evaluate-and-score`.
