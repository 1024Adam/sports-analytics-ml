"""TOI-weighted active skater aggregation for Stage 1 game features (`features/roster.py`).

Weights active skater rolling `ixG/60`, `OnIce-xGD/60`, and `GameScore/60` by
projected TOI share across Forwards (`F`) and Defensemen (`D`), plus
`missing_regular_toi`.
Full implementation scheduled for Milestone 3.
"""

from __future__ import annotations

import polars as pl


def build_roster_strength_features(skater_game_logs: pl.DataFrame) -> pl.DataFrame:
    """Aggregate pre-game active skater rolling metrics weighted by projected TOI share."""
    raise NotImplementedError("Scheduled for Milestone 3 implementation.")
