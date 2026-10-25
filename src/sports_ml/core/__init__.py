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
    SchemaValidationError,
    SportAdapter,
    StarterModifierHook,
    VenueModifierHook,
    validate_canonical_event_frame,
    validate_canonical_participant_frame,
)
from sports_ml.core.ratings import (
    EloConfig,
    FieldSkillConfig,
    MultiEntrantSkillEngine,
    PairwiseEloEngine,
    compute_elo_win_probability,
    compute_mov_multiplier,
)
from sports_ml.core.schedule import (
    compute_schedule_features,
    haversine_distance_km,
    timezone_shift_hours,
)
from sports_ml.core.shrinkage import (
    bayesian_shrinkage,
    compute_regressed_prior,
    shrink_scalar,
)
from sports_ml.core.temporal import (
    pregame_ewma,
    pregame_exposure_rate,
    pregame_rolling_mean,
    pregame_rolling_sum,
    pregame_sample_count,
    sort_events_chronologically,
)
from sports_ml.core.validation import WalkForwardSplit, WalkForwardValidator

__all__ = [
    "CANONICAL_EVENT_KEY_COLS",
    "CANONICAL_PARTICIPANT_KEY_COLS",
    "CANONICAL_TEMPORAL_SORT_COLS",
    "CompetitionTopology",
    "EloConfig",
    "FieldSkillConfig",
    "MultiEntrantSkillEngine",
    "PairwiseEloEngine",
    "SchemaValidationError",
    "SportAdapter",
    "StarterModifierHook",
    "VenueModifierHook",
    "WalkForwardSplit",
    "WalkForwardValidator",
    "bayesian_shrinkage",
    "compute_elo_win_probability",
    "compute_mov_multiplier",
    "compute_regressed_prior",
    "compute_schedule_features",
    "haversine_distance_km",
    "pregame_ewma",
    "pregame_exposure_rate",
    "pregame_rolling_mean",
    "pregame_rolling_sum",
    "pregame_sample_count",
    "shrink_scalar",
    "sort_events_chronologically",
    "timezone_shift_hours",
    "validate_canonical_event_frame",
    "validate_canonical_participant_frame",
]
