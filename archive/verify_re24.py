"""Independently check the RE24 artifacts against source pitches and game labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import parquet


def verify_raw_traces(pitches, predictions):
    raw_path = pitches / "2025/09/statcast_2025_09.parquet"
    columns = [
        "game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot",
        "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score",
        "post_bat_score", "events",
    ]
    raw = pd.read_parquet(raw_path, columns=columns).sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    )
    test = predictions.set_index(["game_pk", "at_bat_number"], verify_integrity=True)
    # Covers runner movement, a scoring sacrifice double play, and a scoring
    # single ending in the batter's third out on the bases.
    for game, at_bat, expected_state in [(776145, 2, 1), (776165, 39, 6), (776220, 37, 7)]:
        play = raw.loc[raw["game_pk"].eq(game)]
        pa = play.loc[play["at_bat_number"].eq(at_bat)]
        first = pa.iloc[0]
        half = play.loc[play["inning"].eq(first.inning) & play["inning_topbot"].eq(first.inning_topbot)]
        start_of_next_half = play.loc[play["at_bat_number"].gt(half["at_bat_number"].max())].iloc[0]
        final_score = start_of_next_half.fld_score
        assert final_score == half.iloc[-1].post_bat_score
        row = test.loc[game, at_bat]
        assert row.bases == expected_state
        assert row.outs == first.outs_when_up
        assert row.runs_remaining == final_score - first.bat_score
        for base in (1, 2, 3):
            runner = first[f"on_{base}b"]
            joined = row[f"runner_{base}"]
            assert pd.isna(joined) if pd.isna(runner) else runner == joined
        if game == 776145:
            assert len(pa) == 6
            assert pd.isna(pa.iloc[-1].on_1b) and pa.iloc[-1].on_2b == first.on_1b

    # A caught stealing ends this half while the batter is at the plate.
    assert pd.isna(raw.loc[raw["game_pk"].eq(776168) & raw["at_bat_number"].eq(62), "events"]).all()
    unfinished = test.loc[776168, 62]
    assert unfinished.outs == 2 and unfinished.bases == 1 and unfinished.runs_remaining == 0
    # The ninth-inning walk-off and tenth-inning runner are ineligible.
    assert not ((predictions.game_pk == 776191) & (predictions.inning == 9)
                & (predictions.inning_topbot == "Bot")).any()
    assert predictions.inning.between(1, 9).all()
    assert ((raw.game_pk == 776147) & raw.inning.eq(10) & raw.on_2b.notna()).any()


def verify_speed(speed_dir, predictions, year):
    lookup = pd.DataFrame(columns=["sprint_speed", "source_year"])
    files = [path for path in speed_dir.glob("*.csv") if path.stem.isdecimal()]
    for path in sorted(files, key=lambda p: (abs(int(p.stem) - year), int(p.stem))):
        speed = pd.read_csv(path).dropna(subset=["player_id", "sprint_speed"])
        speed = speed.set_index("player_id")[["sprint_speed"]]
        speed["source_year"] = int(path.stem)
        lookup = pd.concat([lookup, speed.loc[~speed.index.isin(lookup.index)]])
    for base in (1, 2, 3):
        actual = predictions[f"speed_{base}"]
        runner = predictions[f"runner_{base}"]
        occupied = predictions.bases.astype("int64") & (1 << (base - 1)) != 0
        np.testing.assert_array_equal(occupied, runner.notna())
        assert actual.loc[~occupied].isna().all()
        np.testing.assert_allclose(
            actual.to_numpy(dtype=float, na_value=np.nan),
            runner.map(lookup.sprint_speed).to_numpy(dtype=float, na_value=np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            predictions[f"speed_year_{base}"].to_numpy(dtype=float, na_value=np.nan),
            runner.map(lookup.source_year).to_numpy(dtype=float, na_value=np.nan),
            equal_nan=True,
        )


def read_regular_season(pitches, year, columns, suspended):
    chunks = []
    for file in sorted((pitches / str(year)).glob("*/*.parquet")):
        source = parquet.ParquetFile(file)
        if source.metadata.num_rows:
            month = source.read(columns=["game_type", *columns]).to_pandas()
            selected = month.game_type.eq("R") & ~month.game_pk.isin(suspended)
            chunks.append(month.loc[selected, columns])
    return pd.concat(chunks, ignore_index=True)


def verify_train_matrix(pitches, outputs, suspended):
    columns = [
        "game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot",
        "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score", "batter",
    ]
    stats = []
    for year in (2022, 2023):
        raw = read_regular_season(pitches, year, columns, suspended).sort_values(
            ["game_pk", "at_bat_number", "pitch_number"]
        )
        halves = raw.drop_duplicates(["game_pk", "inning", "inning_topbot"]).copy()
        halves["ordinal"] = 2 * halves.inning + halves.inning_topbot.eq("Bot")
        next_half = halves.groupby("game_pk")[["ordinal", "fld_score"]].shift(-1)
        halves["end_score"] = next_half.fld_score.where(next_half.ordinal.eq(halves.ordinal + 1))
        first = raw.drop_duplicates(["game_pk", "at_bat_number"]).query(
            "pitch_number == 1 and inning <= 9 and outs_when_up in [0, 1, 2]"
        ).dropna(subset=["batter"])
        first = first.merge(halves[["game_pk", "inning", "inning_topbot", "end_score"]],
                            on=["game_pk", "inning", "inning_topbot"], validate="many_to_one")
        first["remaining"] = first.end_score - first.bat_score
        first["bases"] = sum((1 << i) * first[f"on_{i + 1}b"].notna() for i in range(3))
        stats.append(first.loc[first.remaining.ge(0)].groupby(["outs_when_up", "bases"])
                     .remaining.agg(["size", "sum"]))
    counts = stats[0] + stats[1]
    matrix = pd.read_csv(outputs / "train_matrix.csv").sort_values(["outs", "bases"])
    np.testing.assert_array_equal(matrix[["outs", "bases"]], counts.index.to_frame(index=False))
    np.testing.assert_array_equal(matrix.n, counts["size"])
    np.testing.assert_allclose(matrix.re, (counts["sum"] / counts["size"]).to_numpy(dtype=float))


def verify_history(pitches, predictions, suspended):
    # The first test date's batter history comes from earlier seasons.
    row = predictions.loc[predictions.game_date.eq(predictions.game_date.min())].iloc[0]
    columns = [
        "game_date", "game_pk", "at_bat_number", "pitch_number", "batter",
        "events",
    ]
    weights = {
        "walk": .692, "hit_by_pitch": .722, "single": .879, "double": 1.242,
        "triple": 1.568, "home_run": 2.007, "strikeout": 0, "strikeout_double_play": 0,
        "field_out": 0, "force_out": 0, "field_error": 0, "fielders_choice": 0,
        "fielders_choice_out": 0, "grounded_into_double_play": 0, "double_play": 0,
        "triple_play": 0, "sac_fly": 0, "sac_fly_double_play": 0,
    }
    player_counts, league_counts = pd.Series(dtype=float), pd.Series(dtype=float)
    for year in range(2021, int(predictions.game_date.dt.year.iloc[0])):
        season = read_regular_season(pitches, year, columns, suspended)
        ends = season.sort_values("pitch_number").drop_duplicates(
            ["game_pk", "at_bat_number"], keep="last"
        )
        observed = ends.loc[ends.events.isin(weights)]
        league_counts = league_counts.add(observed.events.value_counts(), fill_value=0)
        player = observed.loc[observed.batter.eq(row.batter)]
        player_counts = player_counts.add(player.events.value_counts(), fill_value=0)
    assert player_counts.sum() == row.woba_prior_pa
    values = pd.Series(weights)
    league_mean = league_counts.dot(values.reindex(league_counts.index)) / league_counts.sum()
    weighted_batter = player_counts.dot(values.reindex(player_counts.index))
    np.testing.assert_allclose(row.woba, (weighted_batter + 200 * league_mean) / (row.woba_prior_pa + 200))


def verify_season_scoring(pitches, outputs, predictions):
    scoring = pd.read_csv(outputs / "season_scoring.csv").set_index("season")
    columns = ["game_pk", "home_score", "away_score", "post_home_score", "post_away_score"]
    for season, expected in scoring.iterrows():
        raw = read_regular_season(pitches, season, columns, [])
        home = raw[["home_score", "post_home_score"]].max(axis=1)
        away = raw[["away_score", "post_away_score"]].max(axis=1)
        totals = pd.DataFrame({"home": home, "away": away}).groupby(raw.game_pk).max().sum(axis=1)
        assert len(totals) == expected.games
        assert totals.sum() == expected.total_runs
        np.testing.assert_allclose(totals.mean(), expected.runs_per_game)
    seasonal = pd.read_csv(outputs / "season_re24.csv")
    assert seasonal.groupby("season").size().eq(24).all()
    np.testing.assert_allclose(seasonal.runs_per_game, seasonal.season.map(scoring.runs_per_game))
    np.testing.assert_allclose(predictions.runs_per_game, predictions.season.map(scoring.runs_per_game))
    test_cells = (predictions.groupby(["season", "outs", "bases"]).runs_remaining
                  .agg(n="size", re="mean").reset_index())
    exported = seasonal[seasonal.season.eq(predictions.season.iloc[0])][list(test_cells)]
    pd.testing.assert_frame_equal(test_cells, exported.reset_index(drop=True), check_dtype=False)
    train = seasonal[seasonal.season.isin([2022, 2023])].assign(runs=lambda r: r.n * r.re)
    train = train.groupby(["outs", "bases"])[["n", "runs"]].sum()
    matrix = pd.read_csv(outputs / "train_matrix.csv").set_index(["outs", "bases"])
    np.testing.assert_array_equal(train.n, matrix.n)
    np.testing.assert_allclose(train.runs / train.n, matrix.re)


def verify_outputs(outputs, predictions):
    metadata = json.loads((outputs / "metadata.json").read_text(encoding="utf-8"))
    matrix = pd.read_csv(outputs / "train_matrix.csv")
    assert set(zip(matrix.outs, matrix.bases)) == {(o, b) for o in range(3) for b in range(8)}
    assert len(matrix) == 24 and matrix.n.gt(0).all() and matrix.re.ge(0).all()
    assert matrix.n.sum() == metadata["splits"]["train"]["rows"]
    assert metadata["splits"]["train"]["end"] < metadata["splits"]["validation"]["start"]
    assert metadata["splits"]["validation"]["end"] < metadata["splits"]["test"]["start"]
    expected = predictions.merge(matrix, on=["outs", "bases"], validate="many_to_one")
    np.testing.assert_allclose(expected.pred_m0, expected.re)
    evaluation = pd.read_csv(outputs / "evaluation.csv")
    measured = predictions[["speed_1", "speed_2", "speed_3"]].notna().any(axis=1)
    cohorts = {"all": predictions, "runner_on_measured": predictions.loc[measured]}
    for entry in evaluation.itertuples():
        cohort = cohorts[entry.subset]
        assert len(cohort) == entry.n, entry.subset
        errors = cohort[f"pred_{entry.model.lower()}"] - cohort.runs_remaining
        np.testing.assert_allclose(np.sqrt((errors ** 2).mean()), entry.rmse, rtol=1e-6)
        np.testing.assert_allclose(errors.mean(), entry.mean_error, atol=1e-7)
        assert cohort[f"pred_{entry.model.lower()}"].lt(0).sum() == entry.negative
    calibration = pd.read_csv(outputs / "calibration.csv")
    for (subset, model), cells in calibration.groupby(["subset", "model"]):
        rows = cohorts[subset]
        check = rows.groupby(["outs", "bases"]).agg(
            n=("runs_remaining", "size"), observed=("runs_remaining", "mean"),
            predicted=(f"pred_{model}", "mean"),
        )
        pd.testing.assert_frame_equal(cells.set_index(["outs", "bases"])[list(check)],
                                      check, check_dtype=False, check_index_type=False)
    for entry in pd.read_csv(outputs / "bootstrap.csv").itertuples():
        subset = evaluation.loc[evaluation.subset.eq(entry.subset)].set_index("model")
        names = {name.lower(): value for name, value in subset.rmse.items()}
        np.testing.assert_allclose(entry.delta_rmse, names["m2"] - names["m1"], atol=1e-7)
        assert entry.games == cohorts[entry.subset].game_pk.nunique()
        # Independently reconstruct the clustered sampling distribution.
        sample = cohorts[entry.subset]
        residuals = sample[["pred_m1", "pred_m2"]].subtract(sample.runs_remaining, axis=0)
        per_game = residuals.pow(2).assign(rows=1).groupby(sample.game_pk).sum().to_numpy(dtype=float)
        rng = np.random.default_rng(2048)
        bootstrap_deltas = []
        for _ in range(20):
            games = rng.integers(len(per_game), size=(100, len(per_game)))
            m1, m2, count = per_game[games].sum(axis=1).T
            bootstrap_deltas.extend(np.sqrt(m2 / count) - np.sqrt(m1 / count))
        expected_interval = np.quantile(bootstrap_deltas, [.025, .975])
        tolerance = 0.15 * np.ptp(expected_interval)
        np.testing.assert_allclose([entry.lower, entry.upper], expected_interval, atol=tolerance)
    effects = pd.read_csv(outputs / "effects.csv")
    reference = effects["ordinary_re"]
    np.testing.assert_allclose(effects.difference, effects.adjusted_re - reference)
    beta = np.array(metadata["coefficients"]["m2"])
    state = effects.outs * 8 + effects.bases
    np.testing.assert_allclose(effects.woba_plus_010, beta[24 + state])
    for base in (1, 2, 3):
        empty = effects.bases & (1 << (base - 1)) == 0
        assert effects.loc[empty, f"speed_{base}_plus_1"].isna().all()
        if metadata["m2_choice"] == "pooled":
            np.testing.assert_allclose(effects.loc[~empty, f"speed_{base}_plus_1"], beta[71 + base])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitch-dir", type=Path,
                        default=Path("../../07_Datasets/01_Statcast/raw/pitch_by_pitch"))
    parser.add_argument("--speed-dir", type=Path, default=Path("data/raw/sprint_speed"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--test-year", type=int, choices=(2025, 2026), default=2025)
    args = parser.parse_args()
    suspensions = pd.read_csv(Path(__file__).resolve().parents[1] / "data/reference/suspended_games.csv")
    predictions = pd.read_parquet(args.outputs / "test_predictions.parquet")
    assert not predictions.game_pk.isin(suspensions.game_pk).any()
    assert predictions.game_date.dt.year.eq(args.test_year).all()
    assert predictions.runs_remaining.ge(0).all()
    assert predictions.feature_history_through.lt(predictions.game_date).all()
    assert predictions.groupby("season").runs_per_game.nunique().eq(1).all()
    assert predictions.speed_source_season.eq(args.test_year).all()
    no_history = predictions.woba_prior_pa.eq(0)
    np.testing.assert_allclose(predictions.loc[no_history, "woba"],
                               predictions.loc[no_history, "league_woba"])
    if args.test_year == 2025:
        verify_raw_traces(args.pitch_dir, predictions)
    verify_speed(args.speed_dir, predictions, args.test_year)
    verify_outputs(args.outputs, predictions)
    verify_history(args.pitch_dir, predictions, suspensions.game_pk)
    verify_season_scoring(args.pitch_dir, args.outputs, predictions)
    verify_train_matrix(args.pitch_dir, args.outputs, suspensions.game_pk)
    print("Verified raw PA traces, exclusions, as-of features, all 24 states, and held-out metrics.")


if __name__ == "__main__":
    main()

