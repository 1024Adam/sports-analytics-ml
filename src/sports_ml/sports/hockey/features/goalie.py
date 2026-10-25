"""Starting goalie resolver and rolling shrunk GSAx/60 features (`features/goalie.py`).

Full implementation scheduled for Milestone 3.
"""

from __future__ import annotations

import polars as pl


def build_goalie_rolling_features(goalie_game_logs: pl.DataFrame) -> pl.DataFrame:
    """Compute leak-free rolling shrunk GSAx/60, SV%, and workload fatigue features."""
    raise NotImplementedError("Scheduled for Milestone 3 implementation.")
