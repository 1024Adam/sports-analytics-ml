"""Schedule, B2B, doubleheader, density, and travel tests (`tests/core/test_schedule.py`).

Full B2B, same-day doubleheader, 3-in-4 density, and geodesic distance tests
will be implemented in Milestone 1.
"""

from __future__ import annotations

import pytest

from sports_ml.core.schedule import haversine_distance_km


def test_schedule_stub_signature() -> None:
    """Verify schedule calculator stub raises NotImplementedError prior to Milestone 1."""
    with pytest.raises(NotImplementedError):
        haversine_distance_km(42.3662, -71.0621, 43.6435, -79.3791)
