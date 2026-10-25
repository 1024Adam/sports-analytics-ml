---
name: leak-free-feature-audit
description: "Use when: writing, reviewing, or auditing time-series feature pipelines, rolling/EWMA Polars expressions, Elo/Glicko rating updates, Bayesian shrinkage priors, walk-forward cross-validation splits, or probability calibration to guarantee zero future data leakage."
argument-hint: "Specify the feature module, model pipeline, or Polars transformation to audit for data leakage..."
user-invocable: true
disable-model-invocation: false
---

# Leak-Free Feature & Temporal Audit Playbook

## When to Use
- Whenever implementing or modifying `src/sports_ml/core/temporal.py`, `ratings.py`, `shrinkage.py`, `validation.py`, `calibration.py`, or any `src/sports_ml/sports/<sport>/features/` module.
- When invoked via `/leak-free-feature-audit` to verify that no post-event boxscore or future season information leaks into pre-event predictions.

---

## 1. The Five Temporal Firewall Rules

Every feature and model step MUST pass all five checks:

1. **Strict Chronological Sorting First**:
   Before any `.shift()`, `.rolling_*()`, `.ewm_*()`, or cumulative operation, the Polars DataFrame/LazyFrame MUST be explicitly sorted by:
   ```python
   df = df.sort(["start_time_utc", "event_seq", "event_id"])
   ```
   *(Never sort by `game_date` alone, as MLB doubleheaders and delayed PGA rounds share the same calendar date.)*

2. **Left-Closed Shift (`shift(1)`) Inside Every Partition**:
   Post-event columns (`goals_for`, `xgf_5v5`, `gsax`, `points`, `sog`, `toi_min`, etc.) observed in event $t$ MUST be shifted by `1` within the entity partition (`team`, `player_id`) **before** any rolling aggregation:
   ```python
   # CORRECT (Leak-Free):
   pl.col("xgf_5v5").shift(1).rolling_mean(window_size=10, min_samples=1).over("team")

   # FORBIDDEN (Leaks current game's xGF into pre-game feature!):
   pl.col("xgf_5v5").rolling_mean(window_size=10, min_samples=1).over("team")
   ```

3. **Pre-Event Rating Extraction Before Post-Event Update**:
   In `core/ratings.py`, for any event $e$ at time $T$:
   - First, record `pre_event_rating_home` and `pre_event_rating_away` into the feature table.
   - Second, compute expected outcome $P(\text{Win})$.
   - Third, observe actual score $S$ and update ratings for time $> T$.

4. **Prior & League-Average Baselines Must Be Causally Bounded**:
   - Never compute `df["save_pct"].mean()` across an entire season DataFrame to use as a Bayesian shrinkage prior for Game 5 of that same season.
   - Shrinkage priors must use **prior completed season** baselines or expanding league averages strictly where `completed_time_utc < current_start_time_utc`.

5. **Out-of-Fold Calibration & Walk-Forward Splitting**:
   - Probability calibrators (`Platt` / `Isotonic`) and Stage 2 coherence reconcilers must never be fit on the same fold's training predictions where GBDT trees were fit, and never on test data. Use chronological validation splits (`core/validation.py`).

---

## 2. Step-by-Step Audit Procedure

1. **Scan for Raw Window Calls**:
   Search `src/sports_ml/` for `.rolling_`, `.ewm_`, `.cum_`, `.mean()`, `.sum()`, and `.join(`.
2. **Verify Encapsulation**:
   Confirm that all feature modules in `sports/<sport>/features/` call the guarded helpers in `core/temporal.py` (`pregame_rolling_mean`, `pregame_rolling_sum`, `pregame_ewma`, `pregame_exposure_rate`) rather than writing raw Polars window chains.
3. **Run the Future-Mutation Property Test**:
   Ensure `pytest` includes and passes the canonical mutation test below for the target module.

---

## 3. Canonical Future-Mutation `pytest` Template

Every feature builder `build_features(df)` MUST pass this exact invariance pattern in `tests/`:

```python
import polars as pl
from polars.testing import assert_frame_equal

def test_future_mutation_invariance(sample_chronological_logs: pl.DataFrame) -> None:
    """Mutating events at index >= cutoff_idx must produce zero change in features at < cutoff_idx."""
    baseline_features = build_features(sample_chronological_logs)

    # Pick a midpoint timestamp T_cutoff
    unique_times = baseline_features["start_time_utc"].unique().sort()
    cutoff_time = unique_times[len(unique_times) // 2]

    # Corrupt all post-game stats for events at or after cutoff_time
    corrupted_logs = sample_chronological_logs.with_columns(
        pl.when(pl.col("start_time_utc") >= cutoff_time)
        .then(pl.lit(999.0))
        .otherwise(pl.col(col_name))
        .alias(col_name)
        for col_name in ["goals_for", "goals_against", "xgf_5v5", "xga_5v5"]
        if col_name in sample_chronological_logs.columns
    )

    mutated_features = build_features(corrupted_logs)

    # Filter both feature matrices to events strictly <= cutoff_time
    # (Even the event AT cutoff_time must have identical PRE-GAME features!)
    past_baseline = baseline_features.filter(pl.col("start_time_utc") <= cutoff_time)
    past_mutated = mutated_features.filter(pl.col("start_time_utc") <= cutoff_time)

    assert_frame_equal(past_baseline, past_mutated)
```
