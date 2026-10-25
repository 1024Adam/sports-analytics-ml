"""Team 5v5 process, shot quality, PDO, and special teams rolling features (`features/team.py`).

Computes rolling 5, 10, 25-game and EWMA features for 5v5 xGF%, xGF/60, xGA/60,
CF%, FF%, HD-xGF%, PDO, and Special Teams Net Edge with Bayesian shrinkage.
Full implementation scheduled for Milestone 3.
"""

from __future__ import annotations

import polars as pl


def build_team_rolling_features(team_game_logs: pl.DataFrame) -> pl.DataFrame:
    """Compute leak-free pre-game team process and special-teams rolling features."""
    raise NotImplementedError("Scheduled for Milestone 3 implementation.")
