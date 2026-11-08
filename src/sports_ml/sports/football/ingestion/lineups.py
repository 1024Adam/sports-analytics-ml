"""2-Tier pre-game starting-QB & availability resolver (`sports/football/ingestion/lineups.py`).

* **Tier 1 (official, kickoff - 90 min)**: confirmed starting QB (first dropback QB from
  play-by-play for historical games, or the as-of snapshot in live mode) or the depth chart
  minus the official inactives list.
* **Tier 2 (early-week projection)**: latest depth chart + Wed-Fri injury/practice designations
  -> probabilistic starter (`p_starts`). It **never** reads the `inactives` source.

Every snapshot carries `as_of_time_utc`; a resolver call only reads rows with
`as_of_time_utc <= min(requested as-of, kickoff - 90 min)`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Final, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.sports.football.constants import normalize_football_team

TIER1_CUTOFF_MINUTES: Final[int] = 90

SOURCE_DEPTH_CHART: Final[str] = "depth_chart"
SOURCE_INJURY_REPORT: Final[str] = "injury_report"
SOURCE_INACTIVES: Final[str] = "inactives"
TIER2_SOURCES: Final[tuple[str, ...]] = (SOURCE_DEPTH_CHART, SOURCE_INJURY_REPORT)

# P(player plays | listed healthy) and multipliers by game / practice designation.
HEALTHY_PLAY_PROBABILITY: Final[float] = 0.97
GAME_STATUS_MULTIPLIER: Final[dict[str, float]] = {
    "out": 0.0,
    "doubtful": 0.05,
    "questionable": 0.60,
}
PRACTICE_STATUS_MULTIPLIER: Final[dict[str, float]] = {
    "did not participate": 0.85,
    "limited": 0.95,
    "full": 1.0,
}
MAX_QB_CANDIDATES: Final[int] = 3

QBTier = Literal["tier1", "tier2"]


class QBCandidate(BaseModel):
    """A starting-QB candidate and his probability of starting."""

    model_config = ConfigDict(frozen=True)

    qb_id: int
    p_starts: float


class ResolvedStartingQB(BaseModel):
    """Resolved starting QB for `(game_id, team)` as of a cutoff."""

    model_config = ConfigDict(frozen=True)

    game_id: int
    team: str
    tier: QBTier
    as_of_time_utc: datetime
    starter_id: int | None
    candidates: tuple[QBCandidate, ...]
    source: str


def derive_starting_qbs_from_pbp(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Return `(nflverse_game_id, team, qb_id)` for the first-dropback QB of each team-game.

    nflverse `id` holds the passer (or scrambler) GSIS id on `qb_dropback == 1` plays.
    """
    cols = ["game_id", "play_id", "posteam", "qb_dropback", "id"]
    missing = [c for c in cols if c not in pbp_df.columns]
    if missing:
        raise ValueError(f"Play-by-play frame is missing columns: {missing}.")
    return (
        pbp_df.select(cols)
        .filter(
            (pl.col("qb_dropback") == 1)
            & pl.col("posteam").is_not_null()
            & pl.col("id").is_not_null()
        )
        .sort("game_id", "play_id")
        .group_by("game_id", "posteam", maintain_order=True)
        .agg(pl.col("id").first().alias("gsis_id"))
        .rename({"game_id": "nflverse_game_id", "posteam": "team"})
    )


def _status_multiplier(report_status: str | None, practice_status: str | None) -> float:
    game = (report_status or "").strip().lower()
    if game in GAME_STATUS_MULTIPLIER:
        return GAME_STATUS_MULTIPLIER[game]
    practice = (practice_status or "").strip().lower()
    for token, mult in PRACTICE_STATUS_MULTIPLIER.items():
        if practice.startswith(token):
            return mult
    return 1.0


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class FootballStartingQBResolver:
    """Resolve the starting QB for a team-game using strictly as-of availability snapshots."""

    def __init__(
        self,
        availability: pl.DataFrame,
        games: pl.DataFrame,
        confirmed_starters: pl.DataFrame | None = None,
    ) -> None:
        self._availability = availability
        self._kickoff: dict[int, datetime] = {
            int(row["game_id"]): _utc(row["start_time_utc"])
            for row in games.select("game_id", "start_time_utc").iter_rows(named=True)
        }
        self._confirmed = confirmed_starters

    def kickoff_utc(self, game_id: int) -> datetime:
        """Return the scheduled kickoff instant for `game_id`."""
        if game_id not in self._kickoff:
            raise KeyError(f"Unknown game_id {game_id}.")
        return self._kickoff[game_id]

    def tier1_cutoff(self, game_id: int) -> datetime:
        """Return the Tier 1 information cutoff (`kickoff - 90 min`)."""
        return self.kickoff_utc(game_id) - timedelta(minutes=TIER1_CUTOFF_MINUTES)

    def _as_of_rows(
        self, game_id: int, team: str, sources: tuple[str, ...], as_of: datetime
    ) -> pl.DataFrame:
        return self._availability.filter(
            (pl.col("game_id") == game_id)
            & (pl.col("team") == team)
            & pl.col("source").is_in(list(sources))
            & (pl.col("as_of_time_utc") <= as_of)
        )

    def _depth_qbs(self, rows: pl.DataFrame) -> list[int]:
        depth = rows.filter(
            (pl.col("source") == SOURCE_DEPTH_CHART)
            & (pl.col("position") == "QB")
            & pl.col("depth_rank").is_not_null()
            & pl.col("player_id").is_not_null()
        )
        if depth.is_empty():
            return []
        latest = depth["as_of_time_utc"].max()
        ordered = (
            depth.filter(pl.col("as_of_time_utc") == latest)
            .sort("depth_rank", "player_id")
            .unique(subset=["player_id"], keep="first", maintain_order=True)
        )
        return [int(p) for p in ordered["player_id"].to_list()[:MAX_QB_CANDIDATES]]

    def resolve_tier2(
        self, game_id: int, team: str, as_of_time_utc: datetime
    ) -> ResolvedStartingQB:
        """Early-week probabilistic starter from depth chart + injury/practice reports only."""
        norm_team = normalize_football_team(team)
        effective = min(_utc(as_of_time_utc), self.tier1_cutoff(game_id))
        rows = self._as_of_rows(game_id, norm_team, TIER2_SOURCES, effective)
        qbs = self._depth_qbs(rows)

        injuries = (
            rows.filter(pl.col("source") == SOURCE_INJURY_REPORT)
            .sort("as_of_time_utc")
            .group_by("player_id", maintain_order=True)
            .last()
        )
        status = {
            int(r["player_id"]): _status_multiplier(r["report_status"], r["practice_status"])
            for r in injuries.iter_rows(named=True)
            if r["player_id"] is not None
        }

        weights: list[float] = []
        remaining = 1.0
        for qb in qbs:
            avail = HEALTHY_PLAY_PROBABILITY * status.get(qb, 1.0)
            weights.append(remaining * avail)
            remaining *= 1.0 - avail
        total = sum(weights)
        if qbs and total > 0.0:
            probs = [w / total for w in weights]
        else:
            probs = [1.0 / len(qbs)] * len(qbs) if qbs else []
        candidates = tuple(
            QBCandidate(qb_id=q, p_starts=p) for q, p in zip(qbs, probs, strict=True)
        )
        starter = max(candidates, key=lambda c: c.p_starts).qb_id if candidates else None
        return ResolvedStartingQB(
            game_id=game_id,
            team=norm_team,
            tier="tier2",
            as_of_time_utc=effective,
            starter_id=starter,
            candidates=candidates,
            source="depth_chart+injury_report" if candidates else "none",
        )

    def resolve_tier1(
        self,
        game_id: int,
        team: str,
        as_of_time_utc: datetime | None = None,
    ) -> ResolvedStartingQB:
        """Official starter: confirmed QB, else depth chart minus official inactives.

        Falls back to the Tier 2 projection at the cutoff when no Tier 1 signal exists.
        """
        norm_team = normalize_football_team(team)
        cutoff = self.tier1_cutoff(game_id)
        effective = min(_utc(as_of_time_utc), cutoff) if as_of_time_utc is not None else cutoff

        if self._confirmed is not None and not self._confirmed.is_empty():
            confirmed = self._confirmed.filter(
                (pl.col("game_id") == game_id)
                & (pl.col("team") == norm_team)
                & (pl.col("as_of_time_utc") <= effective)
            ).sort("as_of_time_utc")
            if not confirmed.is_empty():
                qb_id = int(confirmed["qb_id"][-1])
                return ResolvedStartingQB(
                    game_id=game_id,
                    team=norm_team,
                    tier="tier1",
                    as_of_time_utc=effective,
                    starter_id=qb_id,
                    candidates=(QBCandidate(qb_id=qb_id, p_starts=1.0),),
                    source="confirmed_starter",
                )

        rows = self._as_of_rows(
            game_id, norm_team, (SOURCE_DEPTH_CHART, SOURCE_INACTIVES), effective
        )
        inactive_rows = rows.filter(pl.col("source") == SOURCE_INACTIVES)
        qbs = self._depth_qbs(rows)
        if not inactive_rows.is_empty() and qbs:
            inactive_ids = {
                int(p)
                for p in inactive_rows.filter(pl.col("is_inactive"))["player_id"].to_list()
                if p is not None
            }
            active = [q for q in qbs if q not in inactive_ids]
            if active:
                return ResolvedStartingQB(
                    game_id=game_id,
                    team=norm_team,
                    tier="tier1",
                    as_of_time_utc=effective,
                    starter_id=active[0],
                    candidates=(QBCandidate(qb_id=active[0], p_starts=1.0),),
                    source="depth_chart+inactives",
                )
        return self.resolve_tier2(game_id, norm_team, effective)
