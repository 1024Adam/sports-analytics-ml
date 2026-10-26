---
name: evaluate-and-score
description: "Use when: running walk-forward historical backtesting (`sports-ml backtest`) or scoring the live pre-game prediction ledger against completed games (`sports-ml score-predictions`) for `--sport hockey` or `--sport soccer` to evaluate Ranked Probability Score (RPS), Log Loss, Brier Score, Expected Calibration Error (ECE), accuracy, and player projection MAE."
argument-hint: "Specify --sport (hockey|soccer), backtest (--stage 1|2|all, --splits N), or score-predictions..."
user-invocable: true
disable-model-invocation: false
---

# Evaluate & Score Predictions Workflow (`/evaluate-and-score`)

## When to Use
- When the user invokes `/evaluate-and-score` or asks to benchmark model accuracy, run walk-forward cross-validation (`backtest`), or audit real-world live predictions recorded in `data/predictions/` (`score-predictions`).

---

## Step-by-Step Execution Procedure

### Mode A: Walk-Forward Historical Backtest (`sports-ml backtest`)
Simulates training and out-of-sample predicting across chronological folds (`gbdt_train_max_time < calibrator_min_time <= calibrator_max_time < val_min_time`) without touching live network APIs.

1. **Choose Stage (`--stage`) & Folds (`--splits`)**:
   - `--stage 1`: Evaluates Stage 1 calibrated win probabilities against Home Win-Rate, Coin-Flip (`0.50`), and Elo baselines.
   - `--stage all` (or `--stage 2`): Evaluates Stage 1 benchmarks plus Stage 2 skater point projection MAE.
2. **Execute Command**:
   ```bash
   # Backtest Stage 1 and Stage 2 across walk-forward splits (Hockey or Soccer)
   uv run sports-ml backtest --sport hockey --stage all --splits 3
   uv run sports-ml backtest --sport soccer --stage all --splits 3
   ```

### Mode B: Live Prediction Ledger Audit (`sports-ml score-predictions`)
Joins the append-only live prediction ledger (`data/predictions/hockey_predictions.parquet` or `data/predictions/soccer_predictions.parquet`) against subsequently completed `FINAL` games/matches in Silver Parquet tables.

1. **Refresh Completed Games (if needed)**:
   If games predicted earlier have recently finished, sync Silver tables first:
   ```bash
   uv run sports-ml ingest --sport hockey --current-season
   uv run sports-ml ingest --sport soccer --current-season
   ```
2. **Execute Command**:
   ```bash
   uv run sports-ml score-predictions --sport hockey
   uv run sports-ml score-predictions --sport soccer
   ```

### Step 3: Summarize Evaluation Metrics
Report the resulting metrics clearly:
- **Log Loss** and **Brier Score** (lower is better; highlight whether the model beats the Home Win-Rate and Coin-Flip baselines).
- **Expected Calibration Error (ECE)** (measures how closely predicted probabilities match empirical win frequencies).
- **Accuracy** and (when `--stage all` is run) **Stage 2 Skater Points MAE**.
