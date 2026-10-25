"""Sports Analytics ML — Modular, leak-free multi-sport machine learning platform."""

from sports_ml.config import RANDOM_SEED, Settings, get_settings

__version__ = "0.1.0"

__all__ = [
    "RANDOM_SEED",
    "Settings",
    "__version__",
    "get_settings",
]
