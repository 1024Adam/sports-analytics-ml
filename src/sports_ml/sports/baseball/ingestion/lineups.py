"""2-Tier Pre-Game Starting Pitcher, 1-9 Platoon Batting Order & IL Resolver (`lineups.py`).

Tier 1: Official pre-game confirmed 1-9 batting order (`CONFIRMED_LINEUP_1_9`) and
        confirmed/probable starting pitcher (`CONFIRMED_SP` / `PROBABLE_SP`) from
        MLB StatsAPI live game feed or NPB `予告先発` / `スタメン` announcements.
Tier 2: Daytime probable starter rotation resolver (`ROTATION_PROJECTED_SP`) and
        handedness-aware 1-9 platoon batting order (`PLATOON_PROJECTED_LINEUP`)
        conditioned on opposing starting pitcher throwing hand (`opp_sp_hand in {'L', 'R'}`),
        excluding MLB 10/15/60-day IL and NPB deactivated (`ni-gun` farm) players.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from sports_ml.sports.baseball.constants import (
    BaseballLeague,
    normalize_baseball_team,
    resolve_team_league,
)

BaseballStarterStatus = Literal[
    "CONFIRMED_SP",
    "PROBABLE_SP",
    "ROTATION_PROJECTED_SP",
]

BaseballLineupStatus = Literal[
    "CONFIRMED_LINEUP_1_9",
    "PLATOON_PROJECTED_LINEUP",
    "HEURISTIC_LINEUP",
]


class ResolvedBaseballTeamLineup(BaseModel):
    """Pre-game resolved starting pitcher, 1-9 batting order, bench, and IL exclusions."""

    model_config = ConfigDict(frozen=True)

    team: str
    league: BaseballLeague
    starting_pitcher_id: int
    starting_pitcher_hand: Literal["L", "R"]
    starter_status: BaseballStarterStatus
    lineup_status: BaseballLineupStatus
    starting_batter_ids: list[int]
    bench_batter_ids: list[int]
    excluded_unavailable_ids: list[int]

    @property
    def active_batter_ids(self) -> list[int]:
        """All available batters (1-9 starters + bench) eligible for Stage 2 projections."""
        seen: set[int] = set()
        ordered: list[int] = []
        for pid in [*self.starting_batter_ids, *self.bench_batter_ids]:
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        return ordered


class BaseballPreGameLineupResolver:
    """2-Tier waterfall resolver for MLB & NPB starting pitchers and 1-9 batting orders."""

    @staticmethod
    def _extract_team_entry(
        team_code: str,
        payload: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if not payload:
            return None
        direct = payload.get(team_code)
        if isinstance(direct, Mapping):
            return direct
        for side in ("home", "away", "homeTeam", "awayTeam"):
            cand = payload.get(side)
            if isinstance(cand, Mapping):
                c_team = normalize_baseball_team(
                    str(cand.get("team") or cand.get("abbreviation") or "")
                )
                if c_team == team_code:
                    return cand
        return None

    @staticmethod
    def _resolve_starter_from_rotation(
        team_code: str,
        league: BaseballLeague,
        *,
        pitcher_logs_df: pl.DataFrame | None,
        roster_df: pl.DataFrame | None,
        excluded_ids: set[int],
        before_time_utc: datetime | None,
    ) -> tuple[int, Literal["L", "R"]]:
        """Infer today's probable starter from rotation rest cycle (`< before_time_utc`)."""
        if pitcher_logs_df is not None and not pitcher_logs_df.is_empty():
            hist = pitcher_logs_df.filter(
                (pl.col("team") == team_code) & (pl.col("is_starter") == pl.lit(True))
            )
            if before_time_utc is not None and "start_time_utc" in hist.columns:
                hist = hist.filter(pl.col("start_time_utc") < before_time_utc)
            if excluded_ids:
                hist = hist.filter(~pl.col("player_id").is_in(list(excluded_ids)))

            if not hist.is_empty():
                # Target rotation size: 5 starters in MLB, 6 starters in NPB
                rot_size = 6 if league == "NPB" else 5
                summary = (
                    hist.sort("start_time_utc")
                    .group_by("player_id")
                    .agg(
                        pl.len().alias("starts_count"),
                        pl.col("start_time_utc").max().alias("last_start_utc"),
                        pl.col("pitch_hand").last().alias("pitch_hand"),
                    )
                    .sort(["starts_count", "last_start_utc"], descending=[True, False])
                    .head(rot_size)
                )
                # Among the primary rotation arms, select the one furthest back in the cycle
                due_row = summary.sort("last_start_utc", descending=False).row(0, named=True)
                sp_id = int(due_row["player_id"])
                sp_hand_str = str(due_row.get("pitch_hand") or "R").upper()
                sp_hand: Literal["L", "R"] = "L" if sp_hand_str == "L" else "R"
                return (sp_id, sp_hand)

        if roster_df is not None and not roster_df.is_empty():
            avail_p = roster_df.filter(
                (pl.col("team") == team_code)
                & (pl.col("is_pitcher") == pl.lit(True))
                & (~pl.col("player_id").is_in(list(excluded_ids)))
            )
            if not avail_p.is_empty():
                r0 = avail_p.row(0, named=True)
                sp_id = int(r0["player_id"])
                hand_str = str(r0.get("pitch_hand") or "R").upper()
                return (sp_id, "L" if hand_str == "L" else "R")

        return (999_001, "R")

    @staticmethod
    def _resolve_platoon_batting_order(
        team_code: str,
        *,
        opposing_sp_hand: Literal["L", "R"],
        batter_logs_df: pl.DataFrame | None,
        games_df: pl.DataFrame | None,
        roster_df: pl.DataFrame | None,
        excluded_ids: set[int],
        before_time_utc: datetime | None,
    ) -> tuple[list[int], list[int], BaseballLineupStatus]:
        """Project the 1-9 batting order conditioned on `opposing_sp_hand` (`'L'` vs `'R'`)."""
        if batter_logs_df is not None and not batter_logs_df.is_empty():
            hist = batter_logs_df.filter(pl.col("team") == team_code)
            if before_time_utc is not None and "start_time_utc" in hist.columns:
                hist = hist.filter(pl.col("start_time_utc") < before_time_utc)
            if excluded_ids:
                hist = hist.filter(~pl.col("player_id").is_in(list(excluded_ids)))

            if not hist.is_empty():
                # Join opposing starter hand from games_df if available
                if (
                    games_df is not None
                    and not games_df.is_empty()
                    and "home_sp_hand" in games_df.columns
                    and "away_sp_hand" in games_df.columns
                ):
                    g_Lookup = games_df.select(
                        ["game_id", "event_seq", "home_sp_hand", "away_sp_hand"]
                    )
                    hist = hist.join(
                        g_Lookup, on=["game_id", "event_seq"], how="left"
                    ).with_columns(
                        pl.when(pl.col("is_home"))
                        .then(pl.col("away_sp_hand"))
                        .otherwise(pl.col("home_sp_hand"))
                        .fill_null("R")
                        .alias("faced_sp_hand")
                    )
                else:
                    hist = hist.with_columns(pl.lit("R").alias("faced_sp_hand"))

                # Score batters by starts vs this SP hand + total starts + platoon bonus
                agg = (
                    hist.group_by("player_id")
                    .agg(
                        pl.col("is_starter").cast(pl.Int32).sum().alias("total_starts"),
                        (
                            (pl.col("is_starter"))
                            & (pl.col("faced_sp_hand") == pl.lit(opposing_sp_hand))
                        )
                        .cast(pl.Int32)
                        .sum()
                        .alias("same_split_starts"),
                        pl.col("pa").sum().alias("total_pa"),
                        pl.when(pl.col("batting_order_slot") > 0)
                        .then(pl.col("batting_order_slot").cast(pl.Float64))
                        .otherwise(None)
                        .mean()
                        .fill_null(7.5)
                        .alias("avg_slot"),
                        pl.col("bat_side").last().alias("bat_side"),
                    )
                    .with_columns(
                        pl.when(pl.col("bat_side") == pl.lit("S"))
                        .then(pl.lit(1.25))
                        .when(pl.col("bat_side") != pl.lit(opposing_sp_hand))
                        .then(pl.lit(1.50))  # Opposite-hand platoon advantage
                        .otherwise(pl.lit(0.0))
                        .alias("platoon_bonus")
                    )
                    .with_columns(
                        (
                            pl.col("same_split_starts") * 3.0
                            + pl.col("total_starts") * 1.0
                            + pl.col("platoon_bonus")
                            + pl.col("total_pa") * 0.05
                        ).alias("selection_score")
                    )
                    .sort(["selection_score", "player_id"], descending=[True, False])
                )

                top_starters = agg.head(9).sort(["avg_slot", "player_id"])
                starter_ids = [int(pid) for pid in top_starters["player_id"].to_list()]
                bench_ids = [
                    int(pid) for pid in agg["player_id"].to_list() if int(pid) not in starter_ids
                ][:4]
                return (starter_ids, bench_ids, "PLATOON_PROJECTED_LINEUP")

        if roster_df is not None and not roster_df.is_empty():
            avail_b = (
                roster_df.filter(
                    (pl.col("team") == team_code)
                    & (pl.col("is_pitcher") == pl.lit(False))
                    & (~pl.col("player_id").is_in(list(excluded_ids)))
                )
                .with_columns(
                    pl.when(pl.col("bat_side") == pl.lit("S"))
                    .then(pl.lit(1))
                    .when(pl.col("bat_side") != pl.lit(opposing_sp_hand))
                    .then(pl.lit(2))
                    .otherwise(pl.lit(0))
                    .alias("platoon_rank")
                )
                .sort(["platoon_rank", "player_id"], descending=[True, False])
            )
            all_ids = [int(pid) for pid in avail_b["player_id"].to_list()]
            return (all_ids[:9], all_ids[9:13], "HEURISTIC_LINEUP")

        return ([], [], "HEURISTIC_LINEUP")

    @classmethod
    def resolve_team_lineup(
        cls,
        team: str,
        *,
        opposing_sp_hand: str = "R",
        official_lineup_payload: Mapping[str, Any] | None = None,
        probable_sp_id: int | None = None,
        probable_sp_hand: str | None = None,
        roster_df: pl.DataFrame | None = None,
        pitcher_logs_df: pl.DataFrame | None = None,
        batter_logs_df: pl.DataFrame | None = None,
        games_df: pl.DataFrame | None = None,
        before_time_utc: datetime | None = None,
    ) -> ResolvedBaseballTeamLineup:
        """Resolve starting pitcher and 1-9 batting order via the 2-Tier waterfall."""
        team_code = normalize_baseball_team(team)
        league: BaseballLeague = resolve_team_league(team_code) or "MLB"
        opp_hand: Literal["L", "R"] = "L" if str(opposing_sp_hand).upper() == "L" else "R"

        excluded_ids: set[int] = set()
        if roster_df is not None and not roster_df.is_empty():
            team_roster = roster_df.filter(pl.col("team") == team_code)
            if "is_available" in team_roster.columns:
                unavail = team_roster.filter(pl.col("is_available") == pl.lit(False))
                excluded_ids.update(int(pid) for pid in unavail["player_id"].to_list())

        entry = cls._extract_team_entry(team_code, official_lineup_payload)
        if entry is not None:
            for pid in entry.get("excluded_unavailable_ids") or entry.get("il_player_ids") or []:
                excluded_ids.add(int(pid))

        # 1. Resolve Starting Pitcher
        sp_id: int | None = None
        sp_hand: Literal["L", "R"] = "R"
        sp_status: BaseballStarterStatus = "ROTATION_PROJECTED_SP"

        if entry is not None and entry.get("starting_pitcher_id") is not None:
            cand_sp = int(entry["starting_pitcher_id"])
            if cand_sp not in excluded_ids:
                sp_id = cand_sp
                raw_h = str(entry.get("starting_pitcher_hand") or probable_sp_hand or "R").upper()
                sp_hand = "L" if raw_h == "L" else "R"
                sp_status = (
                    "CONFIRMED_SP" if bool(entry.get("is_confirmed_sp", True)) else "PROBABLE_SP"
                )

        if sp_id is None and probable_sp_id is not None and int(probable_sp_id) not in excluded_ids:
            sp_id = int(probable_sp_id)
            sp_hand = "L" if str(probable_sp_hand or "R").upper() == "L" else "R"
            sp_status = "PROBABLE_SP"

        if sp_id is None:
            sp_id, sp_hand = cls._resolve_starter_from_rotation(
                team_code,
                league,
                pitcher_logs_df=pitcher_logs_df,
                roster_df=roster_df,
                excluded_ids=excluded_ids,
                before_time_utc=before_time_utc,
            )
            sp_status = "ROTATION_PROJECTED_SP"

        # 2. Resolve 1-9 Batting Order (Tier 1 Confirmed vs. Tier 2 Platoon-Projected)
        if entry is not None:
            raw_lineup = (
                entry.get("starting_batter_ids")
                or entry.get("batting_order")
                or entry.get("starters")
                or []
            )
            confirmed_batters = [int(bid) for bid in raw_lineup if int(bid) not in excluded_ids][:9]
            if len(confirmed_batters) == 9:
                raw_bench = entry.get("bench_batter_ids") or entry.get("bench") or []
                bench_batters = [
                    int(bid)
                    for bid in raw_bench
                    if int(bid) not in excluded_ids and int(bid) not in confirmed_batters
                ]
                return ResolvedBaseballTeamLineup(
                    team=team_code,
                    league=league,
                    starting_pitcher_id=sp_id,
                    starting_pitcher_hand=sp_hand,
                    starter_status=sp_status,
                    lineup_status="CONFIRMED_LINEUP_1_9",
                    starting_batter_ids=confirmed_batters,
                    bench_batter_ids=bench_batters,
                    excluded_unavailable_ids=sorted(excluded_ids),
                )

        starters_1_9, bench, l_status = cls._resolve_platoon_batting_order(
            team_code,
            opposing_sp_hand=opp_hand,
            batter_logs_df=batter_logs_df,
            games_df=games_df,
            roster_df=roster_df,
            excluded_ids=excluded_ids,
            before_time_utc=before_time_utc,
        )
        return ResolvedBaseballTeamLineup(
            team=team_code,
            league=league,
            starting_pitcher_id=sp_id,
            starting_pitcher_hand=sp_hand,
            starter_status=sp_status,
            lineup_status=l_status,
            starting_batter_ids=starters_1_9,
            bench_batter_ids=bench,
            excluded_unavailable_ids=sorted(excluded_ids),
        )
