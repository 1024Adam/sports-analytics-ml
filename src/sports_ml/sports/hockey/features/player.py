"""Stage 2 individual skater and goalie matchup feature builder (`features/player.py`).

Builds individual rolling EV/PP TOI, vacated injury TOI, per-60 rates, shrunk
career shooting %, opponent defensive suppression, and Stage 1 context signals.
Full implementation scheduled for Milestone 3.
"""

from __future__ import annotations

import polars as pl


def build_stage2_player_features(
    skater_game_logs: pl.DataFrame,
    team_features: pl.DataFrame,
    goalie_features: pl.DataFrame,
) -> pl.DataFrame:
    """Construct pre-game player-matchup features for Stage 2 stat projections."""
    raise NotImplementedError("Scheduled for Milestone 3 implementation.")
