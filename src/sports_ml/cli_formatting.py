"""Formatting helpers for prediction CLI output."""

from __future__ import annotations

from collections.abc import Sequence


def format_outcome_probabilities(probabilities: Sequence[float]) -> tuple[str, ...]:
    """Color the most likely outcome green and all other probabilities yellow."""
    majority = max(probabilities)
    return tuple(
        f"[{'green' if probability == majority else 'yellow'}]{probability:.1%}"
        f"[/{'green' if probability == majority else 'yellow'}]"
        for probability in probabilities
    )
