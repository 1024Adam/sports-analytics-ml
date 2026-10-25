"""Sport-agnostic core ML engine (`src/sports_ml/core/`).

Contains canonical schemas, leak-free Polars temporal expressions, Bayesian
shrinkage, Elo/field rating engines, schedule/travel calculators, walk-forward
splitters, probability calibration, evaluation metrics, and artifact registry.
"""

from sports_ml.core.contracts import (
    CANONICAL_EVENT_KEY_COLS,
    CANONICAL_PARTICIPANT_KEY_COLS,
    CANONICAL_TEMPORAL_SORT_COLS,
    CompetitionTopology,
    SportAdapter,
)

__all__ = [
    "CANONICAL_EVENT_KEY_COLS",
    "CANONICAL_PARTICIPANT_KEY_COLS",
    "CANONICAL_TEMPORAL_SORT_COLS",
    "CompetitionTopology",
    "SportAdapter",
]
