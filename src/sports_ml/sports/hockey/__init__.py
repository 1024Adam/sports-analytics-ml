"""NHL Hockey plugin (`src/sports_ml/sports/hockey/`)."""

from sports_ml.sports.hockey.adapter import HockeyAdapter
from sports_ml.sports.hockey.constants import TRICODE_ALIASES, normalize_tricode

__all__ = [
    "HockeyAdapter",
    "TRICODE_ALIASES",
    "normalize_tricode",
]
