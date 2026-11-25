from __future__ import annotations

import pytest

from sports_ml.cli_formatting import format_outcome_probabilities


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        (
            (0.7, 0.3),
            ("[green]70.0%[/green]", "[yellow]30.0%[/yellow]"),
        ),
        (
            (0.2, 0.55, 0.25),
            ("[yellow]20.0%[/yellow]", "[green]55.0%[/green]", "[yellow]25.0%[/yellow]"),
        ),
        (
            (0.4, 0.4, 0.2),
            ("[green]40.0%[/green]", "[green]40.0%[/green]", "[yellow]20.0%[/yellow]"),
        ),
    ],
)
def test_format_outcome_probabilities_highlights_majority(
    probabilities: tuple[float, ...], expected: tuple[str, ...]
) -> None:
    assert format_outcome_probabilities(probabilities) == expected
