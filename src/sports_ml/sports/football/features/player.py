"""Leak-free Stage 2 player feature matrix (`features/player.py`).

One row per `(game, team, player)` for QB / RB / WR / TE. Every feature uses only the player's
(and opposing defense's) games *strictly before* kickoff: decayed running sums are read to
produce a row and only then updated with that game's outcomes, so mutating the stats of any
game at index `>= t` leaves rows for games `< t` bit-for-bit unchanged.

* Shares and per-opportunity rates are Bayesian-shrunk toward fixed position priors
  (`(n * current + k * prior) / (n + k)`), with draft-capital bonuses for drafted players so
  rookies/debutants never evaluate to NaN.
* `route_participation` is a proxy (decayed snap share): nflverse's free releases carry no
  route data. `wopr = 1.5 * target_share + 0.7 * air_yards_share`.
* `p_active` is the probability the player suits up, from the latest injury-report status /
  inactive designation stamped at or before `min(as_of, kickoff - 90 min)`; missing status
  imputes `1.0` (`p_active_is_imputed`).
* Historical games (with player logs) yield one row per actual participant (targets are the
  realized stats); upcoming games yield rows for the team's recent contributors.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final

import polars as pl

from sports_ml.core.shrinkage import shrink_scalar
from sports_ml.sports.football.adapter import gsis_id_to_int
from sports_ml.sports.football.constants import (
    DRAFT_SHARE_BONUS,
    POSITION_PRIORS,
    STAGE2_SHRINK_K,
    YPRR_PRIORS,
    normalize_football_team,
)
from sports_ml.sports.football.features.player_columns import (
    POSITION_ALIASES,
    STAGE2_IDENTIFIER_COLUMNS,
    STAGE2_PLAYER_FEATURE_COLUMNS,
    STAGE2_POSITIONS,
    STAGE2_TARGET_COLUMNS,
)
from sports_ml.sports.football.features.qb import draft_capital_factor
from sports_ml.sports.football.ingestion.lineups import GAME_STATUS_MULTIPLIER

PLAYER_DECAY: Final[float] = 0.93
TEAM_DECAY: Final[float] = 0.94
CUTOFF_MINUTES: Final[int] = 90
ROSTER_LOOKBACK_GAMES: Final[int] = 4
ROSTER_LIMITS: Final[dict[str, int]] = {"QB": 2, "RB": 4, "WR": 5, "TE": 3}
PRACTICE_STATUS_MULTIPLIER: Final[dict[str, float]] = {
    "did not participate": 0.85,
    "limited": 0.95,
}
QB_SHARE_K: Final[float] = 40.0
QB_TEAM_DECAY: Final[float] = 0.90
LEAGUE_PASS_YDS_PG: Final[float] = 230.0
LEAGUE_RUSH_YDS_PG: Final[float] = 115.0
LEAGUE_REC_YDS_PG: Final[float] = 190.0
LEAGUE_PLAYS_PG: Final[float] = 63.0
LEAGUE_PASS_RATE: Final[float] = 0.58
YPRR_SHRINK_K: Final[float] = 80.0
RYOE_SHRINK_K: Final[float] = 40.0


def normalize_position(raw: object) -> str | None:
    """Map nflverse positions (`HB`, `FB`, ...) onto `QB`/`RB`/`WR`/`TE` (else None)."""
    pos = str(raw or "").strip().upper()
    mapped = POSITION_ALIASES.get(pos, pos)
    return mapped if mapped in STAGE2_POSITIONS else None


@dataclass
class _Player:
    position: str = "WR"
    team: str = ""
    name: str = ""
    career_games: int = 0
    games: float = 0.0
    snap: float = 0.0
    targets: float = 0.0
    rec: float = 0.0
    rec_yds: float = 0.0
    rec_td: float = 0.0
    air: float = 0.0
    rz_t: float = 0.0
    carries: float = 0.0
    rush_yds: float = 0.0
    rush_td: float = 0.0
    gl: float = 0.0
    rz_c: float = 0.0
    att: float = 0.0
    comp: float = 0.0
    pass_yds: float = 0.0
    pass_td: float = 0.0
    ints: float = 0.0
    routes: float = 0.0
    ryoe: float = 0.0
    ryoe_att: float = 0.0
    ngs_seen: int = 0
    t_targets: float = 0.0
    t_air: float = 0.0
    t_rz_t: float = 0.0
    t_carries: float = 0.0
    t_gl: float = 0.0
    t_rz_c: float = 0.0
    t_att: float = 0.0

    def decay(self, g: float) -> None:
        for name in [
            "games",
            "snap",
            "targets",
            "rec",
            "rec_yds",
            "rec_td",
            "air",
            "rz_t",
            "carries",
            "rush_yds",
            "rush_td",
            "gl",
            "rz_c",
            "att",
            "comp",
            "pass_yds",
            "pass_td",
            "ints",
            "routes",
            "ryoe",
            "ryoe_att",
            "t_targets",
            "t_air",
            "t_rz_t",
            "t_carries",
            "t_gl",
            "t_rz_c",
            "t_att",
        ]:
            setattr(self, name, getattr(self, name) * g)


@dataclass
class _Defense:
    games: float = 0.0
    pass_yds: float = 0.0
    rush_yds: float = 0.0
    rec_yds: float = 0.0

    def decay(self, g: float) -> None:
        self.games *= g
        self.pass_yds *= g
        self.rush_yds *= g
        self.rec_yds *= g


@dataclass
class _TeamStyle:
    games: float = 0.0
    plays: float = 0.0
    dropbacks: float = 0.0

    def decay(self, g: float) -> None:
        self.games *= g
        self.plays *= g
        self.dropbacks *= g


@dataclass
class _League:
    games: float = 0.0
    pass_yds: float = 0.0
    rush_yds: float = 0.0
    rec_yds: float = 0.0
    recent: dict[str, deque[dict[int, float]]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=ROSTER_LOOKBACK_GAMES))
    )


def _f(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None else 0.0  # type: ignore[arg-type]


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0.0 else None


def _share_prior(position: str, key: str, draft_factor: float) -> float:
    base = POSITION_PRIORS[position].get(key, 0.0)
    return base + DRAFT_SHARE_BONUS.get(position, {}).get(key, 0.0) * draft_factor


def _p_active_from_snapshots(rows: list[dict[str, object]], cutoff: datetime) -> tuple[float, bool]:
    """Return `(p_active, imputed)` from availability rows stamped at/before `cutoff`."""
    usable = [r for r in rows if r["as_of_time_utc"] <= cutoff]  # type: ignore[operator]
    if not usable:
        return 1.0, True
    if any(r["source"] == "inactives" and r["is_inactive"] for r in usable):
        return 0.0, False
    reports = sorted(
        (r for r in usable if r["source"] == "injury_report"),
        key=lambda r: r["as_of_time_utc"],  # type: ignore[arg-type,return-value]
    )
    if not reports:
        return 1.0, True
    last = reports[-1]
    game = str(last.get("report_status") or "").strip().lower()
    if game in GAME_STATUS_MULTIPLIER:
        return GAME_STATUS_MULTIPLIER[game], False
    practice = str(last.get("practice_status") or "").strip().lower()
    for token, mult in PRACTICE_STATUS_MULTIPLIER.items():
        if practice.startswith(token):
            return mult, False
    return 1.0, False


def _draft_lookup(draft_picks: pl.DataFrame | None) -> dict[int, float]:
    if draft_picks is None or draft_picks.is_empty():
        return {}
    out: dict[int, float] = {}
    for r in draft_picks.iter_rows(named=True):
        raw = r.get("gsis_id")
        pick = r.get("pick")
        if raw is None or pick is None:
            continue
        try:
            out[gsis_id_to_int(raw)] = float(pick)
        except (TypeError, ValueError):
            continue
    return out


def _availability_index(
    availability: pl.DataFrame | None,
) -> dict[tuple[int, str, int], list[dict[str, object]]]:
    index: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    if availability is None or availability.is_empty():
        return index
    wanted = availability.filter(pl.col("source").is_in(["injury_report", "inactives"]))
    for r in wanted.iter_rows(named=True):
        index[(int(r["game_id"]), str(r["team"]), int(r["player_id"]))].append(r)
    return index


def _opp_index(rate: float, league: float) -> float:
    return rate / league if league > 0.0 else 1.0


def build_football_stage2_features(
    games_df: pl.DataFrame,
    player_logs: pl.DataFrame,
    team_logs: pl.DataFrame,
    *,
    availability: pl.DataFrame | None = None,
    draft_picks: pl.DataFrame | None = None,
    stage1_df: pl.DataFrame | None = None,
    as_of_time_utc: datetime | None = None,
    starting_qbs: pl.DataFrame | None = None,
    environment: pl.DataFrame | None = None,
    player_micro: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the Stage 2 Gold matrix (`identifiers + STAGE2_PLAYER_FEATURE_COLUMNS + targets`)."""
    avail = _availability_index(availability)
    draft = _draft_lookup(draft_picks)
    ngs_by: dict[tuple[int, str, int], tuple[float, float]] = {}
    if player_micro is not None and not player_micro.is_empty():
        for r in player_micro.filter(pl.col("ngs_rush_attempts") > 0).iter_rows(named=True):
            ngs_by[(int(r["game_id"]), str(r["team"]), int(r["player_id"]))] = (
                float(r["ngs_ryoe_sum"]),
                float(r["ngs_rush_attempts"]),
            )
    env_penalty: dict[int, float] = {}
    if environment is not None and not environment.is_empty():
        env_penalty = {
            int(r["game_id"]): float(r["env_passing_penalty"])
            for r in environment.select("game_id", "env_passing_penalty").iter_rows(named=True)
        }
    # Resolver-named QBs are always rows for upcoming games (prior-based if they have no history)
    forced_qbs: dict[tuple[int, str], list[int]] = defaultdict(list)
    if starting_qbs is not None:
        for r in starting_qbs.iter_rows(named=True):
            forced_qbs[(int(r["game_id"]), str(r["team"]))].append(int(r["qb_id"]))
    elo_prob: dict[tuple[int, str], float] = {}
    if stage1_df is not None and not stage1_df.is_empty():
        src = "qb_elo_home_win_prob" if "qb_elo_home_win_prob" in stage1_df.columns else None
        src = src or ("elo_home_win_prob" if "elo_home_win_prob" in stage1_df.columns else None)
        if src is not None:
            for r in stage1_df.select("game_id", "home_team", "away_team", src).iter_rows(
                named=True
            ):
                elo_prob[(int(r["game_id"]), str(r["home_team"]))] = float(r[src])
                elo_prob[(int(r["game_id"]), str(r["away_team"]))] = 1.0 - float(r[src])

    positions: dict[int, str | None] = {
        int(r["player_id"]): normalize_position(r["position"])
        for r in player_logs.select("player_id", "position")
        .unique(subset=["player_id"], keep="last")
        .iter_rows(named=True)
    }
    logs = player_logs.filter(pl.col("player_id").is_not_null())
    logs_by: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    team_tot: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in logs.iter_rows(named=True):
        key = (int(r["game_id"]), normalize_football_team(str(r["team"])))
        logs_by[key].append(r)
        t = team_tot[key]
        t["targets"] += _f(r, "targets")
        t["air"] += _f(r, "rec_air_yards")
        t["rz_t"] += _f(r, "rz_targets")
        t["carries"] += _f(r, "carries")
        t["gl"] += _f(r, "gl_carries")
        t["rz_c"] += _f(r, "rz_carries")
        t["att"] += _f(r, "pass_attempts")

    team_style_obs: dict[tuple[int, str], tuple[float, float]] = {}
    for r in team_logs.select("game_id", "team", "off_plays", "off_dropbacks").iter_rows(
        named=True
    ):
        team_style_obs[(int(r["game_id"]), str(r["team"]))] = (
            float(r["off_plays"] or 0.0),
            float(r["off_dropbacks"] or 0.0),
        )

    players: dict[int, _Player] = defaultdict(_Player)
    defenses: dict[str, _Defense] = defaultdict(_Defense)
    styles: dict[str, _TeamStyle] = defaultdict(_TeamStyle)
    team_qb: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    team_qb_total: dict[str, float] = defaultdict(float)
    league = _League()
    out: list[dict[str, object]] = []
    ordered = games_df.sort("start_time_utc", "event_seq", "game_id")

    def defense_features(opp: str) -> tuple[float, float, float]:
        d = defenses[opp]
        k = STAGE2_SHRINK_K["defense_games"]
        lg = max(league.games, 1.0)
        lg_pass, lg_rush, lg_rec = (
            league.pass_yds / lg if league.games else LEAGUE_PASS_YDS_PG,
            league.rush_yds / lg if league.games else LEAGUE_RUSH_YDS_PG,
            league.rec_yds / lg if league.games else LEAGUE_REC_YDS_PG,
        )
        out_idx: list[float] = []
        for total, lg_rate in ((d.pass_yds, lg_pass), (d.rush_yds, lg_rush), (d.rec_yds, lg_rec)):
            rate = _ratio(total, d.games)
            out_idx.append(_opp_index(shrink_scalar(rate, d.games, lg_rate, k), lg_rate))
        return out_idx[0], out_idx[1], out_idx[2]

    def player_row(
        pid: int,
        team: str,
        opp: str,
        is_home: bool,
        g: dict[str, object],
        p_active: float,
        imputed: bool,
        defn: tuple[float, float, float],
        style: tuple[float, float, float],
        elo: float,
    ) -> dict[str, object]:
        st = players[pid]
        pos = st.position
        pri = POSITION_PRIORS[pos]
        dfac = draft_capital_factor(draft.get(pid))
        ks, ky, kt = STAGE2_SHRINK_K["share"], STAGE2_SHRINK_K["yards"], STAGE2_SHRINK_K["td"]
        t_share = shrink_scalar(
            _ratio(st.targets, st.t_targets),
            st.t_targets,
            _share_prior(pos, "target_share", dfac),
            ks,
        )
        air_share = shrink_scalar(
            _ratio(st.air, st.t_air), st.t_air, _share_prior(pos, "air_yards_share", dfac), ks * 4
        )
        carry_share = shrink_scalar(
            _ratio(st.carries, st.t_carries),
            st.t_carries,
            _share_prior(pos, "carry_share", dfac) if pos == "RB" else 0.02,
            ks,
        )
        gl_share = shrink_scalar(
            _ratio(st.gl, st.t_gl),
            st.t_gl,
            pri.get("gl_carry_share", 0.02) if pos == "RB" else 0.02,
            12.0,
        )
        rz_c_share = shrink_scalar(
            _ratio(st.rz_c, st.t_rz_c),
            st.t_rz_c,
            pri.get("rz_carry_share", 0.02) if pos == "RB" else 0.02,
            25.0,
        )
        rz_t_share = shrink_scalar(
            _ratio(st.rz_t, st.t_rz_t), st.t_rz_t, pri.get("rz_target_share", 0.08), 20.0
        )
        qb_total = team_qb_total[team]
        att_share = shrink_scalar(
            _ratio(team_qb[team].get(pid, 0.0), qb_total), qb_total, 0.5, QB_SHARE_K
        )
        snap = shrink_scalar(_ratio(st.snap, st.games), st.games, pri.get("snap_share", 0.5), 4.0)
        catch = (
            shrink_scalar(_ratio(st.rec, st.targets), st.targets, pri["catch_rate"], ky)
            if pos != "QB"
            else 0.0
        )
        ypt = (
            shrink_scalar(_ratio(st.rec_yds, st.targets), st.targets, pri["rec_yds_per_target"], ky)
            if pos != "QB"
            else 0.0
        )
        td_t = (
            shrink_scalar(_ratio(st.rec_td, st.targets), st.targets, pri["rec_td_per_target"], kt)
            if pos != "QB"
            else 0.0
        )
        rush_prior_ypc = POSITION_PRIORS["RB" if pos != "QB" else "QB"]["rush_ypc"]
        rush_prior_td = POSITION_PRIORS["RB" if pos != "QB" else "QB"]["rush_td_per_carry"]
        ypc = shrink_scalar(_ratio(st.rush_yds, st.carries), st.carries, rush_prior_ypc, ky)
        td_c = shrink_scalar(_ratio(st.rush_td, st.carries), st.carries, rush_prior_td, kt)
        qp = POSITION_PRIORS["QB"]
        ka, kq = STAGE2_SHRINK_K["qb_att"], STAGE2_SHRINK_K["qb_td"]
        plays_pg = shrink_scalar(_ratio(style[0], style[2]), style[2], LEAGUE_PLAYS_PG, 6.0)
        pass_rate = shrink_scalar(_ratio(style[1], style[0]), style[0], LEAGUE_PASS_RATE, 300.0)
        row: dict[str, object] = {
            "game_id": int(g["game_id"]),  # type: ignore[call-overload]
            "event_seq": int(g["event_seq"]),  # type: ignore[call-overload]
            "season": int(g["season"]),  # type: ignore[call-overload]
            "season_type": str(g["season_type"]),
            "week": int(g["week"]),  # type: ignore[call-overload]
            "start_time_utc": g["start_time_utc"],
            "game_state": str(g["game_state"]),
            "team": team,
            "opponent": opp,
            "player_id": pid,
            "player_name": st.name,
            "position": pos,
            "is_home_flag": int(is_home),
            "pos_is_qb": int(pos == "QB"),
            "pos_is_rb": int(pos == "RB"),
            "pos_is_wr": int(pos == "WR"),
            "pos_is_te": int(pos == "TE"),
            "p_active": p_active,
            "p_active_is_imputed": int(imputed),
            "is_debut_flag": int(st.career_games == 0),
            "career_games_pre": st.career_games,
            "draft_capital_factor": dfac,
            "snap_share": snap,
            "target_share": t_share if pos != "QB" else 0.0,
            "air_yards_share": air_share if pos != "QB" else 0.0,
            "wopr": (1.5 * t_share + 0.7 * air_share) if pos != "QB" else 0.0,
            "route_participation": snap
            if pos in ("WR", "TE")
            else (snap * 0.5 if pos == "RB" else 0.0),
            "yprr": shrink_scalar(
                _ratio(st.rec_yds, st.routes), st.routes, YPRR_PRIORS[pos], YPRR_SHRINK_K
            )
            if pos in YPRR_PRIORS
            else 0.0,
            "rush_yards_over_expected_per_att": shrink_scalar(
                _ratio(st.ryoe, st.ryoe_att), st.ryoe_att, 0.0, RYOE_SHRINK_K
            )
            if pos in ("RB", "QB")
            else 0.0,
            "ngs_is_imputed": int(st.ngs_seen == 0),
            "rz_target_share": rz_t_share if pos != "QB" else 0.0,
            "carry_share": carry_share if pos in ("RB", "QB") else 0.0,
            "gl_carry_share": gl_share if pos in ("RB", "QB") else 0.0,
            "rz_carry_share": rz_c_share if pos in ("RB", "QB") else 0.0,
            "pass_att_share": att_share if pos == "QB" else 0.0,
            "qb_comp_rate": shrink_scalar(_ratio(st.comp, st.att), st.att, qp["comp_rate"], ka)
            if pos == "QB"
            else 0.0,
            "qb_yds_per_att": shrink_scalar(
                _ratio(st.pass_yds, st.att), st.att, qp["yds_per_att"], ka
            )
            if pos == "QB"
            else 0.0,
            "qb_td_per_att": shrink_scalar(_ratio(st.pass_td, st.att), st.att, qp["td_per_att"], kq)
            if pos == "QB"
            else 0.0,
            "qb_int_per_att": shrink_scalar(_ratio(st.ints, st.att), st.att, qp["int_per_att"], kq)
            if pos == "QB"
            else 0.0,
            "qb_rush_att_per_game": shrink_scalar(
                _ratio(st.carries, st.games), st.games, qp["rush_att_pg"], 6.0
            )
            if pos == "QB"
            else 0.0,
            "qb_rush_ypc": ypc if pos == "QB" else 0.0,
            "qb_rush_td_per_carry": td_c if pos == "QB" else 0.0,
            "rush_ypc": ypc if pos != "QB" else 0.0,
            "rush_td_per_carry": td_c if pos != "QB" else 0.0,
            "catch_rate": catch,
            "rec_yds_per_target": ypt,
            "rec_td_per_target": td_t,
            "opp_def_pass_yds_idx": defn[0],
            "opp_def_rush_yds_idx": defn[1],
            "opp_def_rec_yds_idx": defn[2],
            "team_elo_win_prob": elo,
            "team_pace_plays_pg": plays_pg,
            "team_pass_rate": pass_rate,
            "env_passing_penalty": env_penalty.get(int(g["game_id"]), 0.0),  # type: ignore[call-overload]
        }
        completed = g.get("_logs")
        for col in STAGE2_TARGET_COLUMNS:
            row[col] = None
        if completed is not None:
            lg_row: dict[str, object] = completed  # type: ignore[assignment]
            row.update(
                {
                    "actual_pass_att": _f(lg_row, "pass_attempts"),
                    "actual_completions": _f(lg_row, "completions"),
                    "actual_pass_yds": _f(lg_row, "pass_yards"),
                    "actual_pass_td": _f(lg_row, "pass_tds"),
                    "actual_int": _f(lg_row, "interceptions"),
                    "actual_carries": _f(lg_row, "carries"),
                    "actual_rush_yds": _f(lg_row, "rush_yards"),
                    "actual_rush_td": _f(lg_row, "rush_tds"),
                    "actual_targets": _f(lg_row, "targets"),
                    "actual_rec": _f(lg_row, "receptions"),
                    "actual_rec_yds": _f(lg_row, "rec_yards"),
                    "actual_rec_td": _f(lg_row, "rec_tds"),
                }
            )
        return row

    for g in ordered.iter_rows(named=True):
        gid = int(g["game_id"])
        kickoff: datetime = g["start_time_utc"]
        cutoff = kickoff - timedelta(minutes=CUTOFF_MINUTES)
        if as_of_time_utc is not None:
            cutoff = min(cutoff, as_of_time_utc)
        sides = (
            (str(g["home_team"]), str(g["away_team"]), True),
            (str(g["away_team"]), str(g["home_team"]), False),
        )
        played = [(gid, t) in logs_by for t, _, _ in sides]
        pre_rows: list[dict[str, object]] = []
        for team, opp, is_home in sides:
            defn = defense_features(opp)
            ts = styles[team]
            style = (ts.plays, ts.dropbacks, ts.games)
            elo = elo_prob.get((gid, team), 0.5)
            participants = logs_by.get((gid, team))
            if participants:
                candidates: list[tuple[int, dict[str, object] | None]] = []
                for r in participants:
                    pid = int(r["player_id"])  # type: ignore[call-overload]
                    pos = normalize_position(r.get("position"))
                    if pos is None:
                        continue
                    opp_amt = _f(r, "pass_attempts") + _f(r, "carries") + _f(r, "targets")
                    if opp_amt <= 0.0 and pos != "QB":
                        continue
                    if pos == "QB" and opp_amt <= 0.0:
                        continue
                    players[pid].position = pos
                    players[pid].name = str(r.get("player_name") or players[pid].name)
                    candidates.append((pid, r))
            else:
                candidates = []
                usage: dict[int, float] = defaultdict(float)
                for game_totals in league.recent[team]:
                    for pid, amt in game_totals.items():
                        usage[pid] += amt
                by_pos: dict[str, list[int]] = defaultdict(list)
                for pid, _amt in sorted(usage.items(), key=lambda kv: (-kv[1], kv[0])):
                    st = players[pid]
                    if st.team == team and st.position in ROSTER_LIMITS:
                        by_pos[st.position].append(pid)
                for pos, limit in ROSTER_LIMITS.items():
                    candidates.extend((pid, None) for pid in by_pos[pos][:limit])
                listed = {pid for pid, _ in candidates}
                for qb in forced_qbs.get((gid, team), []):
                    if qb not in listed:
                        players[qb].position = positions.get(qb) or "QB"
                        players[qb].team = team
                        candidates.append((qb, None))
            for pid, log_row in candidates:
                rows = avail.get((gid, team, pid), [])
                p_active, imputed = _p_active_from_snapshots(rows, cutoff)
                gg = dict(g)
                gg["_logs"] = log_row
                pre_rows.append(
                    player_row(pid, team, opp, is_home, gg, p_active, imputed, defn, style, elo)
                )
        out.extend(pre_rows)

        # --- update state strictly after the pre-game rows for both teams were emitted
        if any(played):
            for team, opp, _h in sides:
                if (gid, team) not in logs_by:
                    continue
                tot = team_tot.get((gid, team))
                recent_game: dict[int, float] = {}
                for r in logs_by.get((gid, team), []):
                    pid = int(r["player_id"])  # type: ignore[call-overload]
                    pos = normalize_position(r.get("position")) or positions.get(pid)
                    if pos is None:
                        continue
                    st = players[pid]
                    st.decay(PLAYER_DECAY)
                    st.position, st.team = pos, team
                    snap_raw = r.get("snap_share")
                    st.games += 1.0
                    st.career_games += 1
                    st.snap += (
                        float(snap_raw)  # type: ignore[arg-type]
                        if snap_raw is not None
                        else POSITION_PRIORS[pos].get("snap_share", 0.5)
                    )
                    st.targets += _f(r, "targets")
                    st.rec += _f(r, "receptions")
                    st.rec_yds += _f(r, "rec_yards")
                    st.rec_td += _f(r, "rec_tds")
                    st.air += _f(r, "rec_air_yards")
                    st.rz_t += _f(r, "rz_targets")
                    st.carries += _f(r, "carries")
                    st.rush_yds += _f(r, "rush_yards")
                    st.rush_td += _f(r, "rush_tds")
                    st.gl += _f(r, "gl_carries")
                    st.rz_c += _f(r, "rz_carries")
                    st.att += _f(r, "pass_attempts")
                    st.comp += _f(r, "completions")
                    st.pass_yds += _f(r, "pass_yards")
                    st.pass_td += _f(r, "pass_tds")
                    st.ints += _f(r, "interceptions")
                    if pos in YPRR_PRIORS:
                        snaps = r.get("snaps")
                        plays_obs = team_style_obs.get((gid, team))
                        db_rate = (
                            plays_obs[1] / plays_obs[0]
                            if plays_obs is not None and plays_obs[0] > 0
                            else LEAGUE_PASS_RATE
                        )
                        if snaps is not None:
                            st.routes += float(snaps) * db_rate  # type: ignore[arg-type]
                        elif snap_raw is not None and plays_obs is not None:
                            st.routes += float(snap_raw) * plays_obs[1]  # type: ignore[arg-type]
                    ngs = ngs_by.get((gid, team, pid))
                    if ngs is not None:
                        st.ryoe += ngs[0]
                        st.ryoe_att += ngs[1]
                        st.ngs_seen += 1
                    if tot is not None:
                        st.t_targets += tot["targets"]
                        st.t_air += tot["air"]
                        st.t_rz_t += tot["rz_t"]
                        st.t_carries += tot["carries"]
                        st.t_gl += tot["gl"]
                        st.t_rz_c += tot["rz_c"]
                        st.t_att += tot["att"]
                    amt = _f(r, "pass_attempts") + _f(r, "carries") + _f(r, "targets")
                    if amt > 0:
                        recent_game[pid] = amt
                league.recent[team].append(recent_game)
                for qid in team_qb[team]:
                    team_qb[team][qid] *= QB_TEAM_DECAY
                team_qb_total[team] *= QB_TEAM_DECAY
                for r in logs_by.get((gid, team), []):
                    if normalize_position(r.get("position")) == "QB":
                        team_qb[team][int(r["player_id"])] += _f(r, "pass_attempts")  # type: ignore[call-overload]
                        team_qb_total[team] += _f(r, "pass_attempts")

                d = defenses[opp]
                d.decay(TEAM_DECAY)
                d.games += 1.0
                opp_logs = logs_by.get((gid, team), [])
                d.pass_yds += sum(_f(r, "pass_yards") for r in opp_logs)
                d.rush_yds += sum(_f(r, "rush_yards") for r in opp_logs)
                d.rec_yds += sum(
                    _f(r, "rec_yards")
                    for r in opp_logs
                    if normalize_position(r.get("position")) in ("WR", "TE")
                )
                obs = team_style_obs.get((gid, team))
                if obs is not None:
                    s = styles[team]
                    s.decay(TEAM_DECAY)
                    s.games += 1.0
                    s.plays += obs[0]
                    s.dropbacks += obs[1]
            for team, _opp, _h in sides:
                league.games += 1.0
                league.pass_yds += sum(_f(r, "pass_yards") for r in logs_by.get((gid, team), []))
                league.rush_yds += sum(_f(r, "rush_yards") for r in logs_by.get((gid, team), []))
                league.rec_yds += sum(
                    _f(r, "rec_yards")
                    for r in logs_by.get((gid, team), [])
                    if normalize_position(r.get("position")) in ("WR", "TE")
                )

    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "game_id": pl.Int64,
        "event_seq": pl.Int32,
        "season": pl.Int32,
        "season_type": pl.Utf8,
        "week": pl.Int16,
        "start_time_utc": pl.Datetime("us", "UTC"),
        "game_state": pl.Utf8,
        "team": pl.Utf8,
        "opponent": pl.Utf8,
        "player_id": pl.Int64,
        "player_name": pl.Utf8,
        "position": pl.Utf8,
        **{c: pl.Float64 for c in STAGE2_PLAYER_FEATURE_COLUMNS},
        **{c: pl.Float64 for c in STAGE2_TARGET_COLUMNS},
    }
    for c in (
        "is_home_flag",
        "pos_is_qb",
        "pos_is_rb",
        "pos_is_wr",
        "pos_is_te",
        "p_active_is_imputed",
        "ngs_is_imputed",
        "is_debut_flag",
        "career_games_pre",
    ):
        schema[c] = pl.Int32
    frame = pl.DataFrame(out, schema=schema)
    cols = [*STAGE2_IDENTIFIER_COLUMNS, *STAGE2_PLAYER_FEATURE_COLUMNS, *STAGE2_TARGET_COLUMNS]
    return frame.select(cols).sort("start_time_utc", "event_seq", "game_id", "team", "player_id")
