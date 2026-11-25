"""Build plate-appearance labels and historical player features from Statcast."""

from pathlib import Path

import pandas as pd
from pyarrow import parquet


HISTORY_START = 2021
MODEL_START = 2022
MODEL_END = 2026
PRIOR_PA = 200
HALF = ["game_pk", "half"]
PA = [*HALF, "at_bat_number"]
PITCH = [*PA, "pitch_number"]
COLUMNS = [
    "game_pk", "game_date", "game_type", "inning", "inning_topbot",
    "at_bat_number", "pitch_number", "batter", "outs_when_up", "bat_score",
    "fld_score", "post_bat_score", "home_score", "away_score", "post_home_score",
    "post_away_score", "on_1b", "on_2b", "on_3b", "events", "home_team", "away_team",
]
# FanGraphs Guts constants: unintentional walk, HBP, single, double, triple, HR.
POSITIVE_EVENTS = ("walk", "hit_by_pitch", "single", "double", "triple", "home_run")
SEASON_WEIGHTS = {
    2021: (.692, .722, .879, 1.242, 1.568, 2.007),
    2022: (.689, .720, .884, 1.261, 1.601, 2.072),
    2023: (.696, .726, .883, 1.244, 1.569, 2.004),
    2024: (.689, .720, .882, 1.254, 1.590, 2.050),
    2025: (.691, .722, .882, 1.252, 1.584, 2.037),
}
# Fixed 2021 constants, known before the first prediction in 2022.
# The zero-weight events are official at-bats or sacrifice flies.
WOBA_WEIGHTS = {
    **dict(zip(POSITIVE_EVENTS, SEASON_WEIGHTS[2021])),
    "strikeout": 0., "strikeout_double_play": 0., "field_out": 0.,
    "force_out": 0., "field_error": 0., "fielders_choice": 0.,
    "fielders_choice_out": 0., "grounded_into_double_play": 0.,
    "double_play": 0., "triple_play": 0., "sac_fly": 0.,
    "sac_fly_double_play": 0.,
}
EXCLUDED_EVENTS = {
    "intent_walk", "catcher_interf", "sac_bunt", "sac_bunt_double_play",
    "truncated_pa", "caught_stealing_2b", "caught_stealing_3b",
    "caught_stealing_home", "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home", "other_out",
}


def read_pitches(pitch_dir: Path, end_year: int, history_start: int = HISTORY_START) -> tuple[pd.DataFrame, list[dict]]:
    """Return sorted regular-season pitches, suspended games included, and file coverage."""
    seasons = []
    coverage = []
    for year in range(history_start, end_year + 1):
        parts = []
        for file in sorted((pitch_dir / str(year)).glob("*/*.parquet")):
            source = parquet.ParquetFile(file)
            rows = source.metadata.num_rows
            month = {"file": str(file.relative_to(pitch_dir)), "rows": rows}
            coverage.append(month)
            if not rows:
                continue
            data = source.read(columns=COLUMNS).to_pandas()
            month["game_types"] = {
                str(kind): int(count) for kind, count in data["game_type"].value_counts().items()
            }
            regular = data.loc[data["game_type"].eq("R")].copy()
            if regular.empty:
                continue
            month["regular_games"] = int(regular["game_pk"].nunique())
            parts.append(regular)
        if not parts:
            raise ValueError(f"No regular-season Statcast records for {year}")
        seasons.append(pd.concat(parts, ignore_index=True))

    pitches = pd.concat(seasons, ignore_index=True)
    pitches["game_date"] = pd.to_datetime(pitches["game_date"]).dt.normalize()
    pitches["season"] = pitches["game_date"].dt.year
    pitches["half"] = 2 * (pitches["inning"] - 1) + pitches["inning_topbot"].eq("Bot")
    if pitches[PITCH].isna().any().any() or not pitches["inning_topbot"].isin(["Top", "Bot"]).all():
        raise ValueError("Regular-season pitches contain missing or invalid sequencing keys")
    if pitches.duplicated(PITCH).any():
        raise ValueError("Duplicate game/half/PA/pitch keys in regular-season input")
    return pitches.sort_values(PITCH, ignore_index=True), coverage


def _plate_appearances(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = pitches.drop_duplicates(PA).copy()
    ends = pitches.drop_duplicates(PA, keep="last")

    first = starts.drop_duplicates(HALF)
    last = ends.drop_duplicates(HALF, keep="last")
    # The next batting side's fielding score is the prior batting side's final
    # score, and transition to the next half certifies a completed half-inning.
    next_half = first[[*HALF, "fld_score"]].rename(columns={"fld_score": "final_score"})
    next_half["half"] -= 1
    halves = last[["season", *HALF, "post_bat_score"]].rename(
        columns={"post_bat_score": "terminal_post_bat_score"}
    ).merge(next_half, on=HALF, how="left", validate="one_to_one")
    halves["eligible"] = halves["half"].lt(18) & halves["final_score"].notna()

    starts = starts.merge(
        halves[[*HALF, "final_score", "terminal_post_bat_score", "eligible"]],
        on=HALF, validate="many_to_one",
    )
    starts["runs_remaining"] = starts["final_score"] - starts["bat_score"]
    starts["eligible_half"] = starts["eligible"]
    starts["eligible"] &= (
        starts["pitch_number"].eq(1) & starts["outs_when_up"].between(0, 2)
        & starts["runs_remaining"].ge(0) & starts["batter"].notna()
    )
    starts["eligible"] = starts["eligible"].fillna(False)
    return starts, halves


def _historical_features(
    pitches: pd.DataFrame, result: pd.DataFrame
) -> pd.DataFrame:
    # An inning can end on the bases before the batter completes the PA.
    # Only recorded outcomes in the stated wOBA denominator enter the history.
    outcomes = pitches.drop_duplicates(PA, keep="last")
    unknown = set(outcomes["events"].dropna().unique()) - WOBA_WEIGHTS.keys() - EXCLUDED_EVENTS
    if unknown:
        raise ValueError(f"Unclassified terminal Statcast events: {sorted(unknown)}")
    history = outcomes[["game_date", "batter", "events"]].copy()
    history["weight"] = history["events"].map(WOBA_WEIGHTS)
    history = history.dropna(subset=["weight", "batter"])

    daily_player = history.groupby(["batter", "game_date"], as_index=False).agg(
        weight=("weight", "sum"), denominator=("weight", "size")
    )
    daily_player[["weight", "denominator"]] = daily_player.groupby("batter")[
        ["weight", "denominator"]
    ].cumsum()
    result = pd.merge_asof(
        result.sort_values("game_date"),
        daily_player.sort_values("game_date"),
        by="batter", on="game_date", allow_exact_matches=False,
    ).rename(columns={"denominator": "woba_prior_pa"})

    daily_league = history.groupby("game_date")["weight"].agg(["sum", "count"]).cumsum()
    daily_league["league_woba"] = daily_league["sum"] / daily_league["count"]

    daily = daily_league[["league_woba"]].copy()
    daily["feature_history_through"] = daily.index
    result = pd.merge_asof(
        result, daily.reset_index(), on="game_date", allow_exact_matches=False,
    )
    result["woba_prior_pa"] = result["woba_prior_pa"].fillna(0).astype("int32")
    result["woba"] = (
        result.pop("weight").fillna(0) + PRIOR_PA * result["league_woba"]
    ) / (result["woba_prior_pa"] + PRIOR_PA)
    return result


def _season_scoring(pitches: pd.DataFrame) -> pd.DataFrame:
    """Recover total MLB runs per game from regular-season scoreboards."""
    games = pitches.groupby(["season", "game_pk"])[
        ["home_score", "away_score", "post_home_score", "post_away_score"]
    ].max()
    runs = (
        games[["home_score", "post_home_score"]].max(axis=1)
        + games[["away_score", "post_away_score"]].max(axis=1)
    )
    summary = runs.groupby("season").agg(games="size", total_runs="sum").reset_index()
    summary["runs_per_game"] = summary["total_runs"] / summary["games"]
    return summary


def _join_speed(result: pd.DataFrame, speed_dir: Path) -> tuple[pd.DataFrame, dict]:
    yearly = []
    source_counts = {}
    for path in sorted(speed_dir.glob("*.csv")):
        if not path.stem.isdecimal():
            continue
        season = int(path.stem)
        speed = pd.read_csv(path, usecols=["player_id", "sprint_speed"]).dropna()
        if speed["player_id"].duplicated().any() or not speed["sprint_speed"].between(15, 35).all():
            raise ValueError(f"Malformed Sprint Speed records in {path}")
        source_counts[str(season)] = int(len(speed))
        yearly.append(speed.assign(source_year=season))
    speeds = pd.concat(yearly, ignore_index=True) if yearly else pd.DataFrame(
        columns=["player_id", "sprint_speed", "source_year"]
    )
    closest = pd.concat([
        speeds.assign(speed_source_season=season, gap=(speeds.source_year - season).abs())
        .sort_values(["gap", "source_year"])
        .drop_duplicates("player_id")
        .drop(columns="gap")
        for season in sorted(result["season"].unique())
    ], ignore_index=True)

    result["speed_source_season"] = result["season"]
    result["bases"] = 0
    for base in (1, 2, 3):
        runner = f"runner_{base}"
        result = result.rename(columns={f"on_{base}b": runner})
        result["bases"] += result[runner].notna().astype("int8") * 2 ** (base - 1)
        if speeds.empty:
            result[f"speed_{base}"] = float("nan")
            result[f"speed_year_{base}"] = pd.NA
            continue
        result = result.merge(
            closest.rename(columns={"player_id": runner, "sprint_speed": f"speed_{base}",
                                    "source_year": f"speed_year_{base}"}),
            on=["speed_source_season", runner], how="left", validate="many_to_one",
        )
        result[f"speed_year_{base}"] = result[f"speed_year_{base}"].astype("Int64")
    return result, source_counts


def build_dataset(
    pitches: pd.DataFrame, months: list[dict], speed_dir: Path, end_year: int = MODEL_END,
    *, model_start: int = MODEL_START,
) -> tuple[pd.DataFrame, dict]:
    """Return modeling rows and coverage summaries from read_pitches() output."""
    scoring = _season_scoring(pitches)
    suspended = pd.read_csv(Path(__file__).resolve().parents[1] / "data/reference/suspended_games.csv")
    skipped = suspended[suspended["game_pk"].isin(pitches["game_pk"])]
    pitches = pitches.loc[~pitches["game_pk"].isin(skipped["game_pk"])]
    starts, halves = _plate_appearances(pitches)
    result = starts.loc[starts["eligible"] & starts["season"].between(model_start, end_year)].copy()
    result = _historical_features(pitches, result)
    result = result.merge(scoring[["season", "runs_per_game"]], on="season", validate="many_to_one")
    result, speed_sources = _join_speed(result, Path(speed_dir))
    result = result.rename(columns={"outs_when_up": "outs"})
    audit = {
        "pitch_months": months, "speed_source_rows": speed_sources, "seasons": {},
        "suspended_games_excluded": skipped.to_dict(orient="records"),
        "season_scoring": scoring.to_dict(orient="records"),
    }
    for season, group in starts.groupby("season"):
        year_halves = halves[halves["season"].eq(season)]
        scored_halves = year_halves[year_halves["eligible"]]
        comparable = scored_halves.dropna(subset=["terminal_post_bat_score"])
        audit["seasons"][str(season)] = {
            "games": int(group["game_pk"].nunique()),
            "first_date": str(group["game_date"].min().date()),
            "last_date": str(group["game_date"].max().date()),
            "pa_starts": int(len(group)),
            "eligible_pa_starts": int(group["eligible"].sum()),
            "excluded_pa_in_eligible_half": int((group["eligible_half"] & ~group["eligible"]).sum()),
            "regulation_halves_without_successor_score": int(
                (year_halves["half"].lt(18) & ~year_halves["eligible"]).sum()
            ),
            "regulation_halves": int(year_halves["half"].lt(18).sum()),
            "extra_halves": int(year_halves["half"].ge(18).sum()),
            "next_half_verified_regulation_halves": int(len(scored_halves)),
            "terminal_pitch_score_missing": int(scored_halves["terminal_post_bat_score"].isna().sum()),
            "terminal_pitch_score_gaps": [
                {"game_pk": int(row.game_pk), "half_ordinal": int(row.half),
                 "terminal_score": int(row.terminal_post_bat_score),
                 "verified_score": int(row.final_score)}
                for row in comparable.loc[
                    comparable["terminal_post_bat_score"].ne(comparable["final_score"])
                ].itertuples(index=False)
            ],
            "terminal_pitch_score_disagreements": int(
                comparable["terminal_post_bat_score"].ne(comparable["final_score"]).sum()
            ),
        }
    keep = [
        "game_pk", "game_date", "season", "inning", "inning_topbot", "at_bat_number",
        "pitch_number", "batter", "outs", "bases", "runs_remaining", "woba",
        "runs_per_game", "runner_1", "runner_2", "runner_3", "speed_1", "speed_2",
        "speed_3", "speed_year_1", "speed_year_2", "speed_year_3",
        "bat_score", "final_score", "terminal_post_bat_score",
        "woba_prior_pa", "league_woba", "feature_history_through", "speed_source_season",
    ]
    return result[keep].sort_values(
        ["game_date", "game_pk", "at_bat_number"], ignore_index=True,
    ), audit
