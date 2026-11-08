"""Position-group availability features from as-of injury / inactive / depth-chart snapshots.

For team-game `T` the builder weights every player by his **prior** snap share (mean snap
percentage over the team's last `WINDOW_GAMES` released games) and multiplies by the probability
that he plays, giving six pre-game columns per team:

* `ol_availability_index`, `pass_rush_availability_index`, `secondary_availability_index`,
  `skill_availability_index`: `sum(w * p_play) / sum(w)` over the position group (1.0 = healthy).
* `starter_snap_share_missing`: snap-weighted expected share of regular starters (weight >= 0.5)
  who will miss the game.
* `qb_p_starts`: probability that the depth-chart-designated QB starts (resolver-style).
* `availability_is_imputed`: 1 when no snapshot exists for the team-game or no snap history yet.

`p_play` is 0 for official inactives stamped by the as-of cutoff; otherwise a **historical start
probability** for the player's latest injury designation (`Out`/`Doubtful`/`Questionable`, or a
practice-participation level), estimated as a shrunk empirical rate `P(plays | designation)` from
games that were released strictly before the prediction time, never from `T` or later. Without a
designation the player is assumed to play.

Strict as-of rules: snapshots stamped after `min(kickoff - 90 min, as_of_time_utc)` are ignored,
and a game's outcomes (snap logs, start-probability counts) are released only `RELEASE_DELAY`
after its kickoff and never after `min(kickoff, as_of_time_utc)`.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final

import polars as pl

from sports_ml.sports.football.ingestion.lineups import (
    GAME_STATUS_MULTIPLIER,
    PRACTICE_STATUS_MULTIPLIER,
    SOURCE_DEPTH_CHART,
    SOURCE_INACTIVES,
    SOURCE_INJURY_REPORT,
    TIER1_CUTOFF_MINUTES,
    status_play_probability,
)

WINDOW_GAMES: Final[int] = 6
MIN_WEIGHT: Final[float] = 0.05
STARTER_WEIGHT: Final[float] = 0.5
RELEASE_DELAY: Final[timedelta] = timedelta(hours=4)
PRIOR_STRENGTH: Final[float] = 10.0
MIN_TEAM_SNAP_ROWS: Final[int] = 30
MAX_QB_CANDIDATES: Final[int] = 3

POSITION_GROUPS: Final[dict[str, frozenset[str]]] = {
    "ol": frozenset({"T", "G", "C", "OT", "OG", "OL", "LT", "LG", "RT", "RG"}),
    "pass_rush": frozenset({"DE", "DT", "DL", "NT", "EDGE", "OLB"}),
    "secondary": frozenset({"CB", "S", "SS", "FS", "DB"}),
    "skill": frozenset({"RB", "FB", "HB", "WR", "TE"}),
}
GROUP_SNAP_SIDE: Final[dict[str, str]] = {
    "ol": "offense",
    "skill": "offense",
    "pass_rush": "defense",
    "secondary": "defense",
}

AVAILABILITY_METRICS: Final[tuple[str, ...]] = (
    "ol_availability_index",
    "pass_rush_availability_index",
    "secondary_availability_index",
    "skill_availability_index",
    "starter_snap_share_missing",
    "qb_p_starts",
)
AVAILABILITY_FRAME_SCHEMA: Final[dict[str, pl.DataType]] = {
    "game_id": pl.Int64(),
    "team": pl.Utf8(),
    **{m: pl.Float64() for m in AVAILABILITY_METRICS},
    "availability_is_imputed": pl.Int8(),
    "availability_as_of_time_utc": pl.Datetime("us", "UTC"),
}
AVAILABILITY_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *[f"{side}_{m}" for m in AVAILABILITY_METRICS for side in ("home", "away")],
    *[f"diff_{m}" for m in AVAILABILITY_METRICS],
    "home_availability_is_imputed",
    "away_availability_is_imputed",
)


def injury_designation(report_status: str | None, practice_status: str | None) -> str | None:
    """Return the designation key used for start-probability estimation (None if unlisted)."""
    game = (report_status or "").strip().lower()
    if game in GAME_STATUS_MULTIPLIER:
        return game
    practice = (practice_status or "").strip().lower()
    for token in PRACTICE_STATUS_MULTIPLIER:
        if practice.startswith(token):
            return f"practice:{token}"
    return None


@dataclass
class StartProbabilityTracker:
    """Shrunk empirical `P(plays | designation)` using only outcomes released so far."""

    prior_strength: float = PRIOR_STRENGTH
    counts: dict[str, list[float]] = field(default_factory=lambda: defaultdict(lambda: [0.0, 0.0]))

    def probability(self, designation: str, prior: float) -> float:
        played, total = self.counts.get(designation, [0.0, 0.0])
        return (played + self.prior_strength * prior) / (total + self.prior_strength)

    def update(self, designation: str, played: bool) -> None:
        self.counts[designation][1] += 1.0
        self.counts[designation][0] += float(played)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass
class _Status:
    designations: dict[int, str]
    inactive: set[int]
    qb_depth: list[int]


class _SnapshotIndex:
    """`(game_id, team)` -> as-of sorted snapshot rows (injury report, inactives, QB depth)."""

    def __init__(self, availability: pl.DataFrame | None) -> None:
        self.rows: dict[
            tuple[int, str], list[tuple[datetime, str, int, str | None, str | None, int | None]]
        ] = defaultdict(list)
        if availability is None or availability.is_empty():
            return
        relevant = availability.filter(
            pl.col("source").is_in([SOURCE_INJURY_REPORT, SOURCE_INACTIVES])
            | ((pl.col("source") == SOURCE_DEPTH_CHART) & (pl.col("position") == "QB"))
        ).sort("as_of_time_utc")
        for r in relevant.select(
            "game_id",
            "team",
            "as_of_time_utc",
            "source",
            "player_id",
            "report_status",
            "practice_status",
            "depth_rank",
            "is_inactive",
        ).iter_rows():
            gid, team, stamp, source, pid, rep, prac, rank, inactive = r
            if pid is None:
                continue
            kind = source
            if source == SOURCE_INACTIVES and not inactive:
                continue
            self.rows[(int(gid), str(team))].append(
                (_utc(stamp), kind, int(pid), rep, prac, None if rank is None else int(rank))
            )

    def status_at(self, game_id: int, team: str, cutoff: datetime) -> _Status:
        designations: dict[int, str] = {}
        inactive: set[int] = set()
        depth_by_stamp: dict[datetime, list[tuple[int, int]]] = defaultdict(list)
        for stamp, kind, pid, rep, prac, rank in self.rows.get((game_id, team), []):
            if stamp > cutoff:
                break
            if kind == SOURCE_INJURY_REPORT:
                des = injury_designation(rep, prac)
                if des is None:
                    designations.pop(pid, None)
                else:
                    designations[pid] = des
            elif kind == SOURCE_INACTIVES:
                inactive.add(pid)
            elif rank is not None:
                depth_by_stamp[stamp].append((rank, pid))
        qb_depth: list[int] = []
        if depth_by_stamp:
            latest = depth_by_stamp[max(depth_by_stamp)]
            seen: set[int] = set()
            for _rank, pid in sorted(latest):
                if pid not in seen:
                    seen.add(pid)
                    qb_depth.append(pid)
        return _Status(designations, inactive, qb_depth[:MAX_QB_CANDIDATES])


@dataclass
class _TeamGameSnaps:
    players: dict[int, tuple[str, float, float, bool]]  # pid -> (pos, off_pct, def_pct, played)


def _neutral_row(game_id: int, team: str, cutoff: datetime) -> dict[str, object]:
    row: dict[str, object] = {"game_id": game_id, "team": team}
    row.update({m: 1.0 for m in AVAILABILITY_METRICS if m != "starter_snap_share_missing"})
    row["starter_snap_share_missing"] = 0.0
    row["availability_is_imputed"] = 1
    row["availability_as_of_time_utc"] = cutoff
    return row


def build_availability_features(
    games_df: pl.DataFrame,
    snap_logs: pl.DataFrame | None,
    availability: pl.DataFrame | None,
    *,
    as_of_time_utc: datetime | None = None,
    tracker_out: StartProbabilityTracker | None = None,
) -> pl.DataFrame:
    """Return one `AVAILABILITY_FRAME_SCHEMA` row per `(game_id, team)` (pre-game state only).

    `tracker_out`, when given, receives the final start-probability counts (for inspection).
    """
    ordered = games_df.sort("start_time_utc", "event_seq", "game_id")
    index = _SnapshotIndex(availability)
    snaps: dict[tuple[int, str], _TeamGameSnaps] = defaultdict(lambda: _TeamGameSnaps({}))
    if snap_logs is not None and not snap_logs.is_empty():
        for r in snap_logs.select(
            "game_id",
            "team",
            "player_id",
            "position",
            "offense_snaps",
            "offense_pct",
            "defense_snaps",
            "defense_pct",
        ).iter_rows():
            gid, team, pid, pos, o_snaps, o_pct, d_snaps, d_pct = r
            played = bool((o_snaps or 0) + (d_snaps or 0) > 0)
            snaps[(int(gid), str(team))].players[int(pid)] = (
                str(pos or ""),
                float(o_pct or 0.0),
                float(d_pct or 0.0),
                played,
            )

    tracker = tracker_out if tracker_out is not None else StartProbabilityTracker()
    recent: dict[str, deque[dict[int, tuple[str, float, float]]]] = defaultdict(
        lambda: deque(maxlen=WINDOW_GAMES)
    )
    pending: list[tuple[datetime, int, str, int]] = []  # (release_time, seq, team, game_id)
    seq = 0
    kickoffs = {int(g["game_id"]): _utc(g["start_time_utc"]) for g in ordered.to_dicts()}

    def release(now: datetime) -> None:
        while pending and pending[0][0] <= now:
            _t, _s, team, gid = heapq.heappop(pending)
            roster = snaps.get((gid, team))
            if roster is None or not roster.players:
                continue
            recent[team].append({p: (v[0], v[1], v[2]) for p, v in roster.players.items()})
            if len(roster.players) < MIN_TEAM_SNAP_ROWS:
                continue
            hist = index.status_at(
                gid, team, kickoffs[gid] - timedelta(minutes=TIER1_CUTOFF_MINUTES)
            )
            for pid, des in hist.designations.items():
                played = roster.players.get(pid, ("", 0.0, 0.0, False))[3]
                tracker.update(des, played)

    def p_play(pid: int, status: _Status) -> float:
        if pid in status.inactive:
            return 0.0
        des = status.designations.get(pid)
        if des is None:
            return 1.0
        game = des if des in GAME_STATUS_MULTIPLIER else None
        prior = (
            status_play_probability(game, None)
            if game is not None
            else status_play_probability(None, des.removeprefix("practice:"))
        )
        return tracker.probability(des, prior)

    rows: list[dict[str, object]] = []
    for g in ordered.iter_rows(named=True):
        gid = int(g["game_id"])
        kickoff = _utc(g["start_time_utc"])
        now_eff = min(kickoff, _utc(as_of_time_utc)) if as_of_time_utc is not None else kickoff
        cutoff = kickoff - timedelta(minutes=TIER1_CUTOFF_MINUTES)
        if as_of_time_utc is not None:
            cutoff = min(cutoff, _utc(as_of_time_utc))
        release(now_eff)
        for team in (str(g["home_team"]), str(g["away_team"])):
            window = list(recent[team])
            if not window:
                rows.append(_neutral_row(gid, team, cutoff))
                continue
            status = index.status_at(gid, team, cutoff)
            n = float(len(window))
            pos_of: dict[int, str] = {}
            w_off: dict[int, float] = defaultdict(float)
            w_def: dict[int, float] = defaultdict(float)
            for game_snaps in window:
                for pid, (pos, o_pct, d_pct) in game_snaps.items():
                    if pos:
                        pos_of[pid] = pos.split("/")[0]
                    w_off[pid] += o_pct / n
                    w_def[pid] += d_pct / n
            probs = {pid: p_play(pid, status) for pid in pos_of}
            row: dict[str, object] = {"game_id": gid, "team": team}
            for group, positions in POSITION_GROUPS.items():
                weights = w_off if GROUP_SNAP_SIDE[group] == "offense" else w_def
                num = den = 0.0
                for pid, pos in pos_of.items():
                    w = weights[pid]
                    if pos in positions and w >= MIN_WEIGHT:
                        num += w * probs[pid]
                        den += w
                row[f"{group}_availability_index"] = num / den if den > 0.0 else 1.0
            miss = tot = 0.0
            for pid in pos_of:
                w = max(w_off[pid], w_def[pid])
                if w >= STARTER_WEIGHT:
                    miss += w * (1.0 - probs[pid])
                    tot += w
            row["starter_snap_share_missing"] = miss / tot if tot > 0.0 else 0.0

            qbs = list(status.qb_depth)
            if not qbs:
                leaders = sorted(
                    (pid for pid, pos in pos_of.items() if pos == "QB"),
                    key=lambda p: (-w_off[p], p),
                )
                qbs = leaders[:1]
            if qbs:
                remaining, weights_q = 1.0, []
                for q in qbs:
                    p = p_play(q, status)
                    weights_q.append(remaining * p)
                    remaining *= 1.0 - p
                total = sum(weights_q)
                row["qb_p_starts"] = max(weights_q) / total if total > 0.0 else 0.0
            else:
                row["qb_p_starts"] = 1.0
            row["availability_is_imputed"] = int(not index.rows.get((gid, team)))
            row["availability_as_of_time_utc"] = cutoff
            rows.append(row)
        for team in (str(g["home_team"]), str(g["away_team"])):
            seq += 1
            heapq.heappush(pending, (kickoff + RELEASE_DELAY, seq, team, gid))
    return pl.DataFrame(rows, schema=AVAILABILITY_FRAME_SCHEMA)


def attach_availability_features(base: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Join `home_*`, `away_*`, and `diff_*` availability columns onto a per-game frame."""
    cols = [*AVAILABILITY_METRICS, "availability_is_imputed"]
    out = base
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        part = frame.select("game_id", "team", *cols).rename(
            {"team": team_col, **{c: f"{side}_{c}" for c in cols}}
        )
        out = out.join(part, on=["game_id", team_col], how="left")
    out = out.with_columns(
        *[
            (pl.col(f"home_{m}") - pl.col(f"away_{m}")).alias(f"diff_{m}")
            for m in AVAILABILITY_METRICS
        ]
    )
    fills: list[pl.Expr] = []
    for m in AVAILABILITY_METRICS:
        neutral = 0.0 if m == "starter_snap_share_missing" else 1.0
        fills.extend(pl.col(c).fill_null(neutral) for c in (f"home_{m}", f"away_{m}"))
        fills.append(pl.col(f"diff_{m}").fill_null(0.0))
    fills.extend(
        pl.col(f"{s}_availability_is_imputed").fill_null(1).cast(pl.Int8) for s in ("home", "away")
    )
    return out.with_columns(fills)


__all__ = [
    "AVAILABILITY_FEATURE_COLUMNS",
    "AVAILABILITY_FRAME_SCHEMA",
    "AVAILABILITY_METRICS",
    "POSITION_GROUPS",
    "StartProbabilityTracker",
    "attach_availability_features",
    "build_availability_features",
    "injury_designation",
]
