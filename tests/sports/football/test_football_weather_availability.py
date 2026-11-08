"""Milestone 27 tests: forecast weather, environment penalties, position-group availability."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import pytest

from sports_ml.sports.football.constants import (
    COVERED_ROOF_TYPES,
    SILVER_FOOTBALL_AVAILABILITY_SCHEMA,
    make_football_game_id,
)
from sports_ml.sports.football.features import (
    AVAILABILITY_FEATURE_COLUMNS,
    ENVIRONMENT_FEATURE_COLUMNS,
    STAGE1_FEATURE_COLUMNS,
    StartProbabilityTracker,
    build_availability_features,
    build_environment_features,
    build_football_stage1_features,
    build_football_stage2_features,
    is_forbidden_feature_column,
)
from sports_ml.sports.football.features.environment import cold_penalty, wind_penalty
from sports_ml.sports.football.ingestion import (
    build_availability_snapshots,
    build_games_frame,
    build_snap_game_logs,
)
from sports_ml.sports.football.ingestion.weather import (
    ForecastUnavailableError,
    OpenMeteoForecastClient,
    build_game_weather,
    sync_football_forecasts,
    venue_for_game,
)
from sports_ml.sports.football.models import Stage1FootballGameModel, Stage2FootballPlayerModel
from sports_ml.sports.football.models.stage1_game import LEVEL_MONOTONE_CONSTRAINTS
from tests.sports.football.synthetic import (
    TEAMS,
    SyntheticSilver,
    build_synthetic_player_logs,
    build_synthetic_silver,
)
from tests.sports.football.test_football_ingestion import QB_WAS, _schedule

NOW = datetime(2030, 1, 1, tzinfo=UTC)

# (slot, position, offense_pct, defense_pct) for a 32-player synthetic snap-count roster
ROSTER: list[tuple[str, str, float, float]] = [
    ("OL1", "T", 1.0, 0.0),
    ("OL2", "G", 1.0, 0.0),
    ("OL3", "C", 1.0, 0.0),
    ("OL4", "G", 1.0, 0.0),
    ("OL5", "T", 1.0, 0.0),
    ("OL6", "T", 0.1, 0.0),
    ("DE1", "DE", 0.0, 0.95),
    ("DE2", "DE", 0.0, 0.85),
    ("DT1", "DT", 0.0, 0.8),
    ("DT2", "DT", 0.0, 0.7),
    ("DT3", "DT", 0.0, 0.2),
    ("OLB1", "OLB", 0.0, 0.9),
    ("ILB1", "ILB", 0.0, 0.9),
    ("ILB2", "ILB", 0.0, 0.8),
    ("LB3", "LB", 0.0, 0.3),
    ("CB1", "CB", 0.0, 0.98),
    ("CB2", "CB", 0.0, 0.95),
    ("CB3", "CB", 0.0, 0.9),
    ("CB4", "CB", 0.0, 0.2),
    ("S1", "S", 0.0, 0.99),
    ("S2", "FS", 0.0, 0.95),
    ("S3", "SS", 0.0, 0.3),
    ("QB1", "QB", 1.0, 0.0),
    ("QB2", "QB", 0.0, 0.0),
    ("RB1", "RB", 0.6, 0.0),
    ("RB2", "RB", 0.4, 0.0),
    ("WR1", "WR", 0.9, 0.0),
    ("WR2", "WR", 0.85, 0.0),
    ("WR3", "WR", 0.7, 0.0),
    ("WR4", "WR", 0.2, 0.0),
    ("TE1", "TE", 0.8, 0.0),
    ("TE2", "TE", 0.3, 0.0),
]
SLOT_ORDER = [r[0] for r in ROSTER]


def pid(team: str, slot: str) -> int:
    return 10_000 + TEAMS.index(team) * 100 + SLOT_ORDER.index(slot)


def snap_logs(
    silver: SyntheticSilver,
    *,
    no_snaps: frozenset[tuple[int, int]] = frozenset(),
    only_final: bool = True,
) -> pl.DataFrame:
    """Snap logs per team-game; `no_snaps` lists `(game_id, player_id)` who did not play."""
    rows: list[dict[str, Any]] = []
    games = silver.games.filter(pl.col("game_state") == "FINAL") if only_final else silver.games
    for g in games.iter_rows(named=True):
        for team in (g["home_team"], g["away_team"]):
            for slot, pos, off, dfn in ROSTER:
                p = pid(team, slot)
                out = (int(g["game_id"]), p) in no_snaps
                rows.append(
                    {
                        "game_id": g["game_id"],
                        "event_seq": g["event_seq"],
                        "start_time_utc": g["start_time_utc"],
                        "season": g["season"],
                        "week": g["week"],
                        "team": team,
                        "player_id": p,
                        "position": pos,
                        "offense_snaps": 0 if out else round(off * 65),
                        "offense_pct": off,
                        "defense_snaps": 0 if out else round(dfn * 65),
                        "defense_pct": dfn,
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Int64,
            "event_seq": pl.Int32,
            "start_time_utc": pl.Datetime("us", "UTC"),
            "season": pl.Int32,
            "week": pl.Int16,
            "team": pl.Utf8,
            "player_id": pl.Int64,
            "position": pl.Utf8,
            "offense_snaps": pl.Int16,
            "offense_pct": pl.Float64,
            "defense_snaps": pl.Int16,
            "defense_pct": pl.Float64,
        },
    )


def avail_row(
    game: dict[str, Any],
    team: str,
    player: int,
    source: str,
    stamp: datetime,
    *,
    position: str = "T",
    report: str | None = None,
    practice: str | None = None,
    rank: int | None = None,
    inactive: bool = False,
) -> dict[str, Any]:
    return {
        "as_of_time_utc": stamp,
        "game_id": game["game_id"],
        "season": game["season"],
        "week": game["week"],
        "team": team,
        "player_id": player,
        "position": position,
        "source": source,
        "report_status": report,
        "practice_status": practice,
        "depth_rank": rank,
        "is_inactive": inactive,
    }


def avail_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SILVER_FOOTBALL_AVAILABILITY_SCHEMA)


def team_games(silver: SyntheticSilver, team: str) -> list[dict[str, Any]]:
    return (
        silver.games.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
        .sort("start_time_utc")
        .to_dicts()
    )


def forecast_frame(
    games: pl.DataFrame,
    *,
    wind: float | dict[int, float] = 5.0,
    temp: float = 60.0,
    precip: float = 0.1,
    skip_games: frozenset[int] = frozenset(),
    hours_after: int = 8,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for g in games.iter_rows(named=True):
        venue = venue_for_game(g)
        if venue is None or venue.roof in COVERED_ROOF_TYPES or g["game_id"] in skip_games:
            continue
        w = wind[int(g["game_id"])] if isinstance(wind, dict) else wind
        start = g["start_time_utc"].replace(minute=0, second=0, microsecond=0)
        for h in range(-1, hours_after):
            rows.append(
                {
                    "venue_id": venue.venue_id,
                    "valid_time_utc": start + timedelta(hours=h),
                    "forecast_temp_f": temp,
                    "forecast_wind_mph": w,
                    "forecast_precip_prob": precip,
                    "forecast_lead_hours": 24.0,
                    "source": "previous_day1",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "venue_id": pl.Utf8,
            "valid_time_utc": pl.Datetime("us", "UTC"),
            "forecast_temp_f": pl.Float64,
            "forecast_wind_mph": pl.Float64,
            "forecast_precip_prob": pl.Float64,
            "forecast_lead_hours": pl.Float64,
            "source": pl.Utf8,
        },
    )


@pytest.fixture(scope="module")
def silver() -> SyntheticSilver:
    return build_synthetic_silver(future_weeks=1)


def one_game(silver: SyntheticSilver, home: str) -> pl.DataFrame:
    return silver.games.filter(pl.col("home_team") == home).sort("start_time_utc").head(1)


# ------------------------------------------------------------------------------ weather


def _hourly_payload(
    suffix: str, temp: float, wind: float | None, prob: float | None
) -> dict[str, Any]:
    times = [f"2023-11-05T{h:02d}:00" for h in range(24)]
    return {
        "hourly": {
            "time": times,
            f"temperature_2m{suffix}": [temp] * 24,
            f"wind_speed_10m{suffix}": [wind] * 24,
            f"precipitation_probability{suffix}": [prob] * 24,
        }
    }


def test_forecast_client_parses_caches_retries_and_serves_offline(tmp_path: Path) -> None:
    from datetime import date

    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        if len(hits) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_hourly_payload("_previous_day1", 41.0, 18.0, 70.0))

    client = OpenMeteoForecastClient(
        tmp_path / "wx",
        transport=httpx.MockTransport(handler),
        retry_min_wait=0.0,
    )
    frame, cached = client.fetch_historical(
        "BUF_HIGHMARK", 42.77, -78.79, 2023, date(2023, 11, 5), date(2023, 11, 5)
    )
    assert not cached and len(hits) == 2  # one retry after the 503
    assert frame.height == 24
    assert frame["forecast_wind_mph"][0] == 18.0
    assert frame["forecast_precip_prob"][0] == pytest.approx(0.7)
    assert frame["forecast_lead_hours"][0] == 24.0 and frame["source"][0] == "previous_day1"
    assert "previous_day1" in hits[-1] and "fahrenheit" in hits[-1] and "mph" in hits[-1]

    again, cached = client.fetch_historical(
        "BUF_HIGHMARK", 42.77, -78.79, 2023, date(2023, 11, 5), date(2023, 11, 5)
    )
    assert cached and len(hits) == 2 and again.height == 24

    offline = OpenMeteoForecastClient(tmp_path / "wx", offline=True)
    assert offline.load_cached().height == 24
    with pytest.raises(ForecastUnavailableError):
        offline.fetch_historical(
            "KC_ARROWHEAD", 39.0, -94.4, 2023, date(2023, 11, 5), date(2023, 11, 5)
        )


def test_sync_only_requests_outdoor_venues_and_failures_are_not_fatal(
    tmp_path: Path, silver: SyntheticSilver
) -> None:
    requested: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append((str(request.url.host), request.url.params["latitude"]))
        if "api.open-meteo.com" in str(request.url.host) and "previous" not in str(
            request.url.host
        ):
            return httpx.Response(500)
        return httpx.Response(200, json=_hourly_payload("_previous_day1", 50.0, 6.0, None))

    client = OpenMeteoForecastClient(
        tmp_path / "wx", transport=httpx.MockTransport(handler), retry_attempts=1
    )
    report = sync_football_forecasts(silver.games, client, now=datetime(2024, 3, 1, tzinfo=UTC))
    venues = {
        v.venue_id
        for g in silver.games.iter_rows(named=True)
        if (v := venue_for_game(g)) is not None and v.roof not in COVERED_ROOF_TYPES
    }
    seasons = silver.games["season"].n_unique()
    assert report.windows_requested == len(venues) * seasons
    assert report.windows_failed == 0
    assert all("previous-runs" in host for host, _ in requested)  # domes never requested
    lat_dome = {"36.0909", "33.9535"}
    assert not lat_dome & {lat for _, lat in requested}

    future = silver.games.filter(pl.col("game_state") != "FINAL")
    first_kickoff = future["start_time_utc"].min()
    assert isinstance(first_kickoff, datetime)
    upcoming = sync_football_forecasts(future, client, now=first_kickoff - timedelta(days=3))
    assert (
        upcoming.windows_failed >= 1 and upcoming.failures
    )  # live API 500 -> recorded, not raised


def test_game_weather_uses_forecasts_neutral_indoors_and_flags(silver: SyntheticSilver) -> None:
    fc = forecast_frame(silver.games, wind=22.0, temp=33.0, precip=0.8)
    weather = build_game_weather(silver.games, fc, now=NOW)
    joined = weather.join(silver.games.select("game_id", "home_team"), on="game_id")
    indoor = joined.filter(pl.col("home_team").is_in(["LV", "LAC"]))
    outdoor = joined.filter(~pl.col("home_team").is_in(["LV", "LAC"]))
    assert (indoor["weather_is_indoor"] == 1).all()
    assert (indoor["weather_temp_f"] == 70.0).all() and (indoor["weather_wind_mph"] == 0.0).all()
    assert indoor["forecast_wind_mph"].null_count() == indoor.height
    assert (outdoor["weather_wind_mph"] == 22.0).all() and (outdoor["weather_temp_f"] == 33.0).all()
    assert (outdoor["weather_precip_prob"] == 0.8).all()
    assert (outdoor["forecast_lead_hours"] == 24.0).all()
    assert (outdoor["weather_is_observed_proxy"] == 0).all()
    assert (outdoor["weather_is_imputed"] == 0).all()


def test_observed_proxy_is_flagged_per_season_and_never_mixed(silver: SyntheticSilver) -> None:
    games = silver.games.with_columns(
        pl.when(pl.col("season") == 2021)
        .then(pl.lit(88.0))
        .otherwise(pl.col("obs_weather_temp"))
        .alias("obs_weather_temp"),
        pl.lit(9.5).alias("obs_weather_wind"),
    )
    # 2022: complete forecasts, 2023: forecasts for only half the outdoor games, 2021: none
    half = games.filter(pl.col("season") == 2023).sort("start_time_utc")["game_id"].to_list()
    skip = frozenset(half[::2])
    fc = forecast_frame(games.filter(pl.col("season") != 2021), wind=12.0, skip_games=skip)
    weather = build_game_weather(games, fc, now=NOW).join(
        games.select("game_id", "season", "home_team"), on="game_id"
    )
    outdoor = weather.filter(~pl.col("home_team").is_in(["LV", "LAC"]))

    s2021 = outdoor.filter(pl.col("season") == 2021)
    assert (s2021["weather_is_observed_proxy"] == 1).all()
    assert (s2021["weather_temp_f"] == 88.0).all() and (s2021["weather_wind_mph"] == 9.5).all()
    assert s2021["forecast_temp_f"].null_count() == s2021.height  # forecasts stay forecasts

    s2022 = outdoor.filter(pl.col("season") == 2022)
    assert (s2022["weather_is_observed_proxy"] == 0).all()
    assert (
        s2022["weather_wind_mph"] == 12.0
    ).all()  # observed 9.5 never leaks into a forecast season

    s2023 = outdoor.filter(pl.col("season") == 2023)
    assert (s2023["weather_is_observed_proxy"] == 1).all()  # coverage < 90% -> whole-season proxy
    assert (s2023["weather_wind_mph"] == 9.5).all()
    assert s2023["forecast_wind_mph"].drop_nulls().len() > 0  # present but deliberately unused


def test_weather_features_ignore_observed_columns_and_out_of_window_forecasts(
    silver: SyntheticSilver,
) -> None:
    fc = forecast_frame(silver.games, wind=11.0, hours_after=10)
    base = build_game_weather(silver.games, fc, now=NOW)
    target = silver.games.filter(pl.col("game_state") == "FINAL").sort("start_time_utc")[40]
    kickoff = target["start_time_utc"][0]

    mutated_games = silver.games.with_columns(
        (pl.col("obs_weather_temp") - 40.0).alias("obs_weather_temp"),
        (pl.col("obs_weather_wind") + 30.0).alias("obs_weather_wind"),
        pl.lit("closed").alias("obs_weather_roof"),
    )
    late = pl.col("valid_time_utc") >= kickoff.replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=4)
    mutated_fc = fc.with_columns(
        pl.when(late)
        .then(pl.col("forecast_wind_mph") + 25.0)
        .otherwise(pl.col("forecast_wind_mph"))
        .alias("forecast_wind_mph"),
        pl.when(late)
        .then(pl.col("forecast_temp_f") - 30.0)
        .otherwise(pl.col("forecast_temp_f"))
        .alias("forecast_temp_f"),
    )
    changed = build_game_weather(mutated_games, mutated_fc, now=NOW)
    gid = target["game_id"][0]
    assert base.filter(pl.col("game_id") == gid).equals(changed.filter(pl.col("game_id") == gid))
    # the venue-hour window still moves when the in-window forecast moves (control)
    moved = fc.with_columns(
        pl.when(~late)
        .then(pl.col("forecast_wind_mph") + 25.0)
        .otherwise(pl.col("forecast_wind_mph"))
        .alias("forecast_wind_mph")
    )
    assert not build_game_weather(silver.games, moved, now=NOW).equals(base)


# ----------------------------------------------------------------------------- environment


def test_wind_forecast_lowers_passing_environment_monotonically_and_not_in_domes(
    silver: SyntheticSilver,
) -> None:
    outdoor = one_game(silver, "BUF")
    dome = one_game(silver, "LV")
    games = pl.concat([outdoor, dome])
    winds = [0.0, 10.0, 15.0, 20.0, 25.0, 35.0]
    outdoor_pen, dome_pen, shifts, totals = [], [], [], []
    for w in winds:
        fc = forecast_frame(games, wind=w, temp=65.0, precip=0.0)
        env = build_environment_features(games, build_game_weather(games, fc, now=NOW))
        by_home = env.join(games.select("game_id", "home_team"), on="game_id")
        o = by_home.filter(pl.col("home_team") == "BUF").row(0, named=True)
        d = by_home.filter(pl.col("home_team") == "LV").row(0, named=True)
        outdoor_pen.append(o["env_passing_penalty"])
        shifts.append(o["env_total_shift"])
        totals.append(o["wind_penalty"])
        dome_pen.append(
            (
                d["env_passing_penalty"],
                d["env_kicking_penalty"],
                d["env_total_shift"],
                d["wind_penalty"],
            )
        )
    assert outdoor_pen[:3] == [0.0, 0.0, 0.0]  # nothing at or below 15 mph
    assert all(a < b for a, b in zip(outdoor_pen[2:], outdoor_pen[3:], strict=False))
    assert all(a >= b for a, b in zip(shifts, shifts[1:], strict=False)) and shifts[-1] < shifts[2]
    assert totals[-1] > totals[3] > 0.0
    assert set(dome_pen) == {(0.0, 0.0, 0.0, 0.0)}  # domes ignore the wind entirely
    # convexity above the threshold: each extra 10 mph costs more than the last
    assert (
        wind_penalty(np.array([35.0]))[0] - wind_penalty(np.array([25.0]))[0]
        > wind_penalty(np.array([25.0]))[0] - wind_penalty(np.array([15.0]))[0]
    )
    assert cold_penalty(np.array([40.0, 30.0, 20.0])).tolist() == [0.0, 1.0, 2.0]


def test_cold_flags_open_roof_damping_and_surface_interactions(silver: SyntheticSilver) -> None:
    buf = one_game(silver, "BUF")  # outdoors, turf; MIA visits (warm-weather team)
    fc = forecast_frame(buf, wind=20.0, temp=25.0, precip=0.7)
    env = build_environment_features(buf, build_game_weather(buf, fc, now=NOW)).row(0, named=True)
    assert env["away_warm_team_in_cold"] == 1.0 and env["home_warm_team_in_cold"] == 0.0
    assert env["diff_warm_team_in_cold"] == -1.0
    assert env["is_precipitation_risk"] == 1.0 and env["roof_exposure"] == 1.0
    assert env["is_turf"] == 1.0 and env["cold_penalty_x_turf"] == pytest.approx(
        env["cold_penalty"]
    )
    assert env["precip_risk_x_grass"] == 0.0

    warm = buf.with_columns(pl.lit("MIA").alias("home_team"), pl.lit("NE").alias("away_team"))
    warm_env = build_environment_features(warm, build_game_weather(warm, fc, now=NOW)).row(
        0, named=True
    )
    assert warm_env["precip_risk_x_grass"] in (0.0, 1.0)

    open_roof = buf.with_columns(pl.lit("MAD_BERNABEU").alias("venue_id"))
    fc2 = forecast_frame(open_roof, wind=20.0, temp=25.0, precip=0.7)
    damped = build_environment_features(open_roof, build_game_weather(open_roof, fc2, now=NOW)).row(
        0, named=True
    )
    assert damped["roof_exposure"] == 0.5
    assert damped["wind_penalty"] == pytest.approx(0.5 * wind_penalty(np.array([20.0]))[0])
    assert 0.0 < damped["env_passing_penalty"] < env["env_passing_penalty"]


def test_environment_columns_satisfy_the_odds_firewall() -> None:
    assert not [c for c in ENVIRONMENT_FEATURE_COLUMNS if is_forbidden_feature_column(c)]
    assert not [c for c in AVAILABILITY_FEATURE_COLUMNS if is_forbidden_feature_column(c)]
    assert set(ENVIRONMENT_FEATURE_COLUMNS) <= set(STAGE1_FEATURE_COLUMNS)
    assert set(AVAILABILITY_FEATURE_COLUMNS) <= set(STAGE1_FEATURE_COLUMNS)


def test_stage1_monotone_constraints_are_signed_for_environment_levels() -> None:
    model = Stage1FootballGameModel()
    model.feature_columns = ["diff_qb_value", "wind_penalty", "cold_penalty", "env_total_shift"]
    assert model._monotone(True) == [1, -1, -1, 1]
    assert model._monotone(False) == [-1, -1, -1, 1]
    assert LEVEL_MONOTONE_CONSTRAINTS["env_passing_penalty"] == -1


# ----------------------------------------------------------------------------- availability


@pytest.fixture(scope="module")
def snaps(silver: SyntheticSilver) -> pl.DataFrame:
    return snap_logs(silver)


def _target(silver: SyntheticSilver, team: str = "BUF", n: int = 6) -> dict[str, Any]:
    return team_games(silver, team)[n]


def _features(
    silver: SyntheticSilver,
    snaps: pl.DataFrame,
    availability: pl.DataFrame | None,
    **kw: object,
) -> pl.DataFrame:
    return build_availability_features(silver.games, snaps, availability, **kw)  # type: ignore[arg-type]


def _row(frame: pl.DataFrame, game: dict[str, Any], team: str) -> dict[str, Any]:
    return frame.filter((pl.col("game_id") == game["game_id"]) & (pl.col("team") == team)).row(
        0, named=True
    )


def test_healthy_team_has_unit_indices_and_cold_start_is_imputed(
    silver: SyntheticSilver, snaps: pl.DataFrame
) -> None:
    frame = _features(silver, snaps, None)
    first = team_games(silver, "BUF")[0]
    assert _row(frame, first, "BUF")["availability_is_imputed"] == 1
    row = _row(frame, _target(silver), "BUF")
    assert (
        row["availability_is_imputed"] == 1
    )  # no snapshots at all: QB falls back to the snap leader
    assert row["qb_p_starts"] == 1.0 and row["starter_snap_share_missing"] == 0.0
    for m in ("ol", "pass_rush", "secondary", "skill"):
        assert row[f"{m}_availability_index"] == 1.0


def test_losing_a_starter_lowers_only_his_group_in_proportion_to_prior_snap_share(
    silver: SyntheticSilver, snaps: pl.DataFrame
) -> None:
    game = _target(silver)
    cutoff = game["start_time_utc"] - timedelta(minutes=90)

    def inactive(slot: str) -> pl.DataFrame:
        return avail_frame(
            [avail_row(game, "BUF", pid("BUF", slot), "inactives", cutoff, inactive=True)]
        )

    lt = _row(_features(silver, snaps, inactive("OL1")), game, "BUF")
    backup = _row(_features(silver, snaps, inactive("OL6")), game, "BUF")
    rusher = _row(_features(silver, snaps, inactive("DE1")), game, "BUF")

    ol_total = 5 * 1.0 + 0.1
    assert lt["ol_availability_index"] == pytest.approx(1.0 - 1.0 / ol_total)
    assert backup["ol_availability_index"] == pytest.approx(1.0 - 0.1 / ol_total)
    assert (1.0 - lt["ol_availability_index"]) / (
        1.0 - backup["ol_availability_index"]
    ) == pytest.approx(10.0)
    rush_total = 0.95 + 0.85 + 0.8 + 0.7 + 0.2 + 0.9
    assert rusher["pass_rush_availability_index"] == pytest.approx(1.0 - 0.95 / rush_total)
    for other in ("secondary", "skill"):
        assert (
            rusher[f"{other}_availability_index"] == 1.0
            and lt[f"{other}_availability_index"] == 1.0
        )
    assert lt["pass_rush_availability_index"] == 1.0 and rusher["ol_availability_index"] == 1.0
    starters = [max(o, d) for _s, _p, o, d in ROSTER if max(o, d) >= 0.5]
    assert lt["starter_snap_share_missing"] == pytest.approx(1.0 / sum(starters))
    assert lt["availability_as_of_time_utc"] == cutoff


def test_questionable_start_probability_uses_only_previously_released_games(
    silver: SyntheticSilver,
) -> None:
    games = team_games(silver, "BUF")
    target = games[6]
    q = pid("BUF", "OL3")
    rows = [
        avail_row(
            g,
            "BUF",
            q,
            "injury_report",
            g["start_time_utc"] - timedelta(days=2),
            report="Questionable",
        )
        for g in games[:8]
    ]
    avail = avail_frame(rows)
    base_snaps = snap_logs(silver)
    tracker = StartProbabilityTracker()
    row = _row(_features(silver, base_snaps, avail, tracker_out=tracker), target, "BUF")

    prior, k = 0.6, 10.0
    assert row["ol_availability_index"] == pytest.approx(
        1.0 - (1.0 - (6 + k * prior) / (6 + k)) / 5.1
    )
    assert tracker.counts["questionable"][1] >= 6.0

    future_dnp = frozenset((int(g["game_id"]), q) for g in games[6:])
    mutated = _row(_features(silver, snap_logs(silver, no_snaps=future_dnp), avail), target, "BUF")
    assert mutated == row  # outcomes of games >= T cannot move T's features

    past_dnp = frozenset({(int(games[2]["game_id"]), q)})
    earlier = _row(_features(silver, snap_logs(silver, no_snaps=past_dnp), avail), target, "BUF")
    assert (
        earlier["ol_availability_index"] < row["ol_availability_index"]
    )  # control: past outcomes do count
    expected = 1.0 - (1.0 - (5 + k * prior) / (6 + k)) / 5.1
    assert earlier["ol_availability_index"] == pytest.approx(expected)

    assert StartProbabilityTracker().probability("questionable", 0.6) == pytest.approx(0.6)


def test_qb_p_starts_follows_the_depth_chart_and_injury_designation(
    silver: SyntheticSilver, snaps: pl.DataFrame
) -> None:
    game = _target(silver)
    kick = game["start_time_utc"]
    depth = [
        avail_row(
            game,
            "BUF",
            pid("BUF", "QB1"),
            "depth_chart",
            kick - timedelta(days=3),
            position="QB",
            rank=1,
        ),
        avail_row(
            game,
            "BUF",
            pid("BUF", "QB2"),
            "depth_chart",
            kick - timedelta(days=3),
            position="QB",
            rank=2,
        ),
    ]
    healthy = _row(_features(silver, snaps, avail_frame(depth)), game, "BUF")
    doubtful = _row(
        _features(
            silver,
            snaps,
            avail_frame(
                [
                    *depth,
                    avail_row(
                        game,
                        "BUF",
                        pid("BUF", "QB1"),
                        "injury_report",
                        kick - timedelta(days=2),
                        position="QB",
                        report="Doubtful",
                    ),
                ]
            ),
        ),
        game,
        "BUF",
    )
    assert healthy["qb_p_starts"] == pytest.approx(1.0) and healthy["availability_is_imputed"] == 0
    assert doubtful["qb_p_starts"] == pytest.approx(0.95)  # QB1 0.05, QB2 plays 0.95 of the time


def test_availability_is_strictly_as_of(silver: SyntheticSilver, snaps: pl.DataFrame) -> None:
    game = _target(silver)
    kick = game["start_time_utc"]
    as_of = kick - timedelta(days=2)
    baseline_rows = [
        avail_row(
            g,
            team,
            pid(team, "QB1"),
            "depth_chart",
            g["start_time_utc"] - timedelta(days=3),
            position="QB",
            rank=1,
        )
        for g in silver.games.iter_rows(named=True)
        for team in (g["home_team"], g["away_team"])
    ]
    base_avail = avail_frame(baseline_rows)
    later = [
        *[
            avail_row(
                game,
                "BUF",
                pid("BUF", slot),
                "inactives",
                kick - timedelta(minutes=90),
                inactive=True,
            )
            for slot in ("OL1", "OL2", "DE1", "CB1")
        ],
        avail_row(
            game,
            "BUF",
            pid("BUF", "QB1"),
            "injury_report",
            as_of + timedelta(hours=2),
            position="QB",
            report="Out",
        ),
        avail_row(
            game,
            "BUF",
            pid("BUF", "QB1"),
            "depth_chart",
            as_of + timedelta(hours=5),
            position="QB",
            rank=9,
        ),
    ]
    mutated_avail = pl.concat([base_avail, avail_frame(later)])

    a = _features(silver, snaps, base_avail, as_of_time_utc=as_of)
    b = _features(silver, snaps, mutated_avail, as_of_time_utc=as_of)
    assert a.equals(b)  # bit-for-bit: nothing stamped after the as-of time was read
    assert _row(a, game, "BUF")["availability_as_of_time_utc"] == as_of

    unrestricted = _features(silver, snaps, mutated_avail)
    assert _row(unrestricted, game, "BUF")["ol_availability_index"] < 1.0  # control: rows matter


def test_availability_ignores_outcomes_after_the_as_of_time(
    silver: SyntheticSilver, snaps: pl.DataFrame
) -> None:
    games = team_games(silver, "BUF")
    target, previous = games[6], games[5]
    q = pid("BUF", "OL3")
    avail = avail_frame(
        [
            avail_row(
                g,
                "BUF",
                q,
                "injury_report",
                g["start_time_utc"] - timedelta(days=2),
                report="Questionable",
            )
            for g in games[:7]
        ]
    )
    # as-of lands before the previous game finished: its outcome cannot be used
    as_of = previous["start_time_utc"] + timedelta(hours=1)
    no_show = snap_logs(silver, no_snaps=frozenset({(int(previous["game_id"]), q)}))
    a = _row(_features(silver, snaps, avail, as_of_time_utc=as_of), target, "BUF")
    b = _row(_features(silver, no_show, avail, as_of_time_utc=as_of), target, "BUF")
    assert a["ol_availability_index"] == b["ol_availability_index"]


# --------------------------------------------------------------- Gold / Stage 2 integration


def test_gold_includes_environment_and_availability_without_future_leakage(
    silver: SyntheticSilver,
) -> None:
    snaps_all = snap_logs(silver)
    avail = avail_frame(
        [
            avail_row(
                g,
                team,
                pid(team, "OL1"),
                "injury_report",
                g["start_time_utc"] - timedelta(days=2),
                report="Questionable",
            )
            for g in silver.games.iter_rows(named=True)
            for team in (g["home_team"], g["away_team"])
        ]
    )
    fc = forecast_frame(silver.games, wind=17.0, hours_after=10)
    weather = build_game_weather(silver.games, fc, now=NOW)
    gold = build_football_stage1_features(
        silver.games,
        silver.team_logs,
        silver.qb_logs,
        availability=avail,
        snap_logs=snaps_all,
        weather=weather,
    )
    new_cols = [*ENVIRONMENT_FEATURE_COLUMNS, *AVAILABILITY_FEATURE_COLUMNS]
    assert set(new_cols) <= set(gold.columns)
    for c in new_cols:
        assert gold[c].null_count() == 0, c
    assert (
        gold.select(
            (
                pl.col("diff_ol_availability_index")
                - (pl.col("home_ol_availability_index") - pl.col("away_ol_availability_index"))
            )
            .abs()
            .max()
        ).item()
        < 1e-12
    )
    assert float(gold["wind_penalty"].max()) > 0.0  # type: ignore[arg-type]

    target = silver.games.filter(pl.col("game_state") == "FINAL").sort("start_time_utc")[60]
    t0, gid = target["start_time_utc"][0], int(target["game_id"][0])
    later_ids = silver.games.filter(pl.col("start_time_utc") > t0)["game_id"].to_list()
    mutated_snaps = snaps_all.with_columns(
        pl.when(pl.col("start_time_utc") >= t0)
        .then(pl.col("offense_pct") * 0.1)
        .otherwise(pl.col("offense_pct"))
        .alias("offense_pct"),
        pl.when(pl.col("start_time_utc") >= t0)
        .then(pl.lit(0))
        .otherwise(pl.col("offense_snaps"))
        .cast(pl.Int16)
        .alias("offense_snaps"),
    )
    mutated_avail = avail.with_columns(
        pl.when(pl.col("game_id").is_in(later_ids))
        .then(pl.lit("Out"))
        .otherwise(pl.col("report_status"))
        .alias("report_status")
    )
    mutated_weather = weather.with_columns(
        pl.when(pl.col("game_id").is_in(later_ids))
        .then(pl.lit(40.0))
        .otherwise(pl.col("weather_wind_mph"))
        .alias("weather_wind_mph")
    )
    mutated_games = silver.games.with_columns(
        pl.lit(77.0).alias("obs_weather_wind"), pl.lit(-9.0).alias("obs_weather_temp")
    )
    changed = build_football_stage1_features(
        mutated_games,
        silver.team_logs,
        silver.qb_logs,
        availability=mutated_avail,
        snap_logs=mutated_snaps,
        weather=mutated_weather,
    )
    a = gold.filter(pl.col("game_id") == gid).select(new_cols)
    b = changed.filter(pl.col("game_id") == gid).select(new_cols)
    assert a.equals(b)


def test_default_gold_weather_is_neutral_and_flagged_imputed(silver: SyntheticSilver) -> None:
    gold = build_football_stage1_features(silver.games, silver.team_logs, silver.qb_logs)
    outdoor = gold.join(silver.games.select("game_id", "home_team"), on="game_id").filter(
        ~pl.col("home_team").is_in(["LV", "LAC"])
    )
    assert (outdoor["weather_is_imputed"] == 1).all() and (outdoor["wind_penalty"] == 0.0).all()
    assert (gold["home_availability_is_imputed"] == 1).all()  # no snap logs supplied


def test_stage2_passing_yards_fall_with_the_environment_penalty(silver: SyntheticSilver) -> None:
    players = build_synthetic_player_logs(silver)
    gold = build_football_stage1_features(silver.games, silver.team_logs, silver.qb_logs)
    stage1 = Stage1FootballGameModel(min_calibration_samples=30).fit(
        gold.filter(pl.col("game_state") == "FINAL")
    )
    preds = stage1.predict(gold)
    fc = forecast_frame(silver.games, wind=28.0, temp=25.0, precip=0.9)
    env = build_environment_features(silver.games, build_game_weather(silver.games, fc, now=NOW))
    rows = build_football_stage2_features(
        silver.games, players, silver.team_logs, stage1_df=gold, environment=env
    )
    assert float(rows["env_passing_penalty"].max()) > 0.0  # type: ignore[arg-type]
    model = Stage2FootballPlayerModel().fit(rows, preds)

    upcoming = rows.filter(pl.col("game_state") != "FINAL")
    outdoors = upcoming.filter(pl.col("env_passing_penalty") > 0.0)
    assert not outdoors.is_empty()
    game = int(outdoors["game_id"][0])
    team = str(outdoors.filter(pl.col("game_id") == game)["team"][0])
    sub = upcoming.filter((pl.col("game_id") == game) & (pl.col("team") == team))
    calm = sub.with_columns(pl.lit(0.0).alias("env_passing_penalty"))

    def passing(frame: pl.DataFrame) -> tuple[float, float]:
        out = model.predict(frame, preds, with_probabilities=False)
        return (
            float(out.filter(pl.col("position") == "QB")["proj_pass_yds"].sum()),
            float(out.filter(pl.col("position").is_in(["WR", "TE"]))["proj_rec_yds"].sum()),
        )

    stormy_qb, stormy_rec = passing(sub)
    calm_qb, calm_rec = passing(calm)
    assert stormy_qb < calm_qb and stormy_rec < calm_rec
    assert passing(sub.with_columns(pl.lit(1.0).alias("env_passing_penalty")))[0] > stormy_qb


# ------------------------------------------------------------------------ Silver tables


def test_snap_logs_keep_all_positions_and_depth_charts_keep_all_formations() -> None:
    games = build_games_frame(_schedule())
    gid = make_football_game_id(2023, 1, "ARI", "WAS")
    snaps_raw = pl.DataFrame(
        {
            "game_id": ["2023_01_ARI_WAS"] * 3,
            "season": [2023] * 3,
            "week": [1] * 3,
            "team": ["WAS"] * 3,
            "position": ["QB", "T", "CB"],
            "pfr_player_id": ["P_QB", "P_LT", "P_CB"],
            "offense_snaps": [60.0, 60.0, 0.0],
            "offense_pct": [0.97, 0.97, 0.0],
            "defense_snaps": [0.0, 0.0, 55.0],
            "defense_pct": [0.0, 0.0, 0.9],
        }
    )
    players = pl.DataFrame(
        {
            "gsis_id": [QB_WAS, "00-0031111", "00-0032222"],
            "pfr_id": ["P_QB", "P_LT", "P_CB"],
        }
    )
    logs = build_snap_game_logs(snaps_raw, players, games)
    assert sorted(logs["position"].to_list()) == ["CB", "QB", "T"]
    assert set(logs["game_id"].to_list()) == {gid}
    assert logs.filter(pl.col("position") == "CB")["defense_pct"][0] == 0.9
    assert build_snap_game_logs(None, players, games).is_empty()

    legacy_depth = pl.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "club_code": ["WAS", "WAS"],
            "gsis_id": [QB_WAS, "00-0032222"],
            "depth_team": ["1", "1"],
            "position": ["QB", "CB"],
            "formation": ["Offense", "Defense"],
        }
    )
    snapshots = build_availability_snapshots(games, depth_charts_df=legacy_depth)
    assert sorted(snapshots["position"].to_list()) == ["CB", "QB"]

    modern_depth = pl.DataFrame(
        {
            "dt": ["2023-09-06T12:00:00Z", "2023-09-06T12:00:00Z"],
            "team": ["WAS", "WAS"],
            "gsis_id": [QB_WAS, "00-0031111"],
            "pos_abb": ["QB", "LT"],
            "pos_rank": [1, 1],
        }
    )
    modern = build_availability_snapshots(games, depth_charts_df=modern_depth)
    assert modern["game_id"].unique().to_list() == [gid]
    assert sorted(modern["position"].to_list()) == ["LT", "QB"]
    kickoff = games.filter(pl.col("game_id") == gid)["start_time_utc"][0]
    assert (modern["as_of_time_utc"] < kickoff).all()
    assert modern.filter(pl.col("position") == "QB")["depth_rank"][0] == 1
