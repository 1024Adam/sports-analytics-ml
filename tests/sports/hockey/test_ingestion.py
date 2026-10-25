"""NHL ingestion, tricode normalization, and Silver reconciliation tests (`test_ingestion.py`).

Full NHL Web API, MoneyPuck CSV, 2-Tier lineup, and Silver upsert tests
will be implemented in Milestone 2.
"""

from __future__ import annotations

from sports_ml.core.contracts import SportAdapter
from sports_ml.sports.hockey import HockeyAdapter, normalize_tricode


def test_hockey_adapter_and_tricode_normalization() -> None:
    """Verify HockeyAdapter satisfies SportAdapter protocol and normalizes legacy tricodes."""
    adapter = HockeyAdapter()
    assert isinstance(adapter, SportAdapter)
    assert adapter.sport_slug == "hockey"
    assert adapter.topology == "head_to_head"

    assert normalize_tricode("S.J") == "SJS"
    assert normalize_tricode("L.A") == "LAK"
    assert normalize_tricode("N.J") == "NJD"
    assert normalize_tricode("T.B") == "TBL"
    assert normalize_tricode("ARI") == "UTA"
    assert normalize_tricode("bos") == "BOS"
