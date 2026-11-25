"""Fit RE24 with positive wOBA slopes over 3.75–5.00 team runs per game."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import lsq_linear

from dataset import build_dataset, read_pitches
from re24_interactions import N_COLUMNS, RPG, RUNNER, RUNNER_RPG, WOBA, WOBA_RPG, design

RPG_LOW, RPG_HIGH, RPG_UNIT = 3.75, 5.00, .25
FIRST_SEASON, VALID_SEASON = 2021, 2024
MIN_WOBA_EFFECT = 1e-6
# endpoint_design() reuses the WOBA and WOBA_RPG columns as the wOBA slope
# at RPG_LOW and at RPG_HIGH.
WOBA_LOW = WOBA
WOBA_HIGH = WOBA_RPG


def endpoint_design(rows, rpg_reference, speed_reference):
    x = design(rows, rpg_reference=rpg_reference, speed_reference=speed_reference)
    if x.shape[1] != N_COLUMNS:
        raise ValueError("Unexpected interaction design layout")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in design matrix")
    team_rpg = rows.runs_per_game.to_numpy() / 2
    centered_rpg = team_rpg - rpg_reference
    np.testing.assert_allclose(x[:, RPG].sum(axis=1), centered_rpg, atol=1e-12)
    np.testing.assert_allclose(x[:, RUNNER_RPG], centered_rpg * x[:, RUNNER].sum(axis=1), atol=1e-12)
    # Express the same straight-line wOBA slope using its two endpoint values.
    high_weight = (team_rpg - RPG_LOW) / (RPG_HIGH - RPG_LOW)
    if not np.all((high_weight >= 0) & (high_weight <= 1)):
        raise ValueError("Team RPG is outside the constrained range")
    woba = x[:, WOBA_LOW].copy()
    x[:, WOBA_LOW] = woba * (1 - high_weight[:, None])
    x[:, WOBA_HIGH] = woba * high_weight[:, None]
    x[:, RPG] /= RPG_UNIT
    x[:, RUNNER_RPG] /= RPG_UNIT
    return x


def without_woba_interaction(x):
    reduced = x[:, :WOBA_HIGH.start].copy()
    reduced[:, WOBA_LOW] += x[:, WOBA_HIGH]
    return np.column_stack((reduced, x[:, RUNNER_RPG]))


def fit_models(train, valid):
    rpg_reference = train.runs_per_game.mean() / 2
    speed_reference = np.nanmean(train[["speed_1", "speed_2", "speed_3"]].to_numpy())
    # QR of [X, y] preserves the least-squares objective without forming X.T @ X.
    triangular = np.empty((0, N_COLUMNS + 1))
    for start in range(0, len(train), 50000):
        batch = train.iloc[start:start + 50000]
        x = endpoint_design(batch, rpg_reference, speed_reference)
        y = batch.runs_remaining.to_numpy()
        if not np.isfinite(y).all():
            raise ValueError("Non-finite runs remaining")
        augmented = np.column_stack((x, y))
        triangular = qr(np.vstack((triangular, augmented)), mode="r")[0][:N_COLUMNS + 1]
    reduced, target = triangular[:, :-1], triangular[:, -1]
    lower = np.full(reduced.shape[1], -np.inf)
    lower[WOBA_LOW] = lower[WOBA_HIGH] = MIN_WOBA_EFFECT
    fit = lsq_linear(reduced, target, bounds=(lower, np.inf), method="bvls")
    if not fit.success:
        raise RuntimeError(fit.message)
    unconstrained = np.linalg.lstsq(reduced, target, rcond=None)[0]
    constant_woba = np.linalg.lstsq(without_woba_interaction(reduced), target, rcond=None)[0]

    x = endpoint_design(valid, rpg_reference, speed_reference)
    predictions = {
        "constant_woba_slope": without_woba_interaction(x) @ constant_woba,
        "unconstrained": x @ unconstrained,
        "positive_woba": x @ fit.x,
    }
    return predictions, fit, unconstrained, rpg_reference, speed_reference


def metrics(predictions, valid):
    y = valid.runs_remaining.to_numpy()
    return [(name, len(valid), np.sqrt(np.mean((prediction - y) ** 2)),
             np.count_nonzero(prediction < 0)) for name, prediction in predictions.items()]


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def bootstrap(rows, rng, repeats=2000):
    """Paired game-level bootstrap of RMSE(pred_m2) - RMSE(pred_m1)."""
    errors = rows.assign(
        sq_m1=(rows.pred_m1 - rows.runs_remaining)**2,
        sq_m2=(rows.pred_m2 - rows.runs_remaining)**2,
        n=1,
    ).groupby("game_pk")[["sq_m1", "sq_m2", "n"]].sum().to_numpy()
    draws = []
    for _ in range(repeats):
        first, second, count = errors[rng.integers(len(errors), size=len(errors))].sum(axis=0)
        draws.append(np.sqrt(second / count) - np.sqrt(first / count))
    return {
        "games": len(errors),
        "delta_rmse": rmse(rows.pred_m2 - rows.runs_remaining) - rmse(rows.pred_m1 - rows.runs_remaining),
        "lower": np.quantile(draws, .025),
        "upper": np.quantile(draws, .975),
    }


def load_rows(project, pitch_dir, monthly, rounded, first_season=FIRST_SEASON, last_season=VALID_SEASON):
    """Return modeling rows for first_season–last_season, the league-month table (or None), and the pitches."""
    # The season before first_season is read only for as-of history features.
    pitches, months = read_pitches(pitch_dir, end_year=last_season, history_start=first_season - 1)
    rows, _ = build_dataset(pitches, months, project / "data/raw/sprint_speed",
                            end_year=last_season, model_start=first_season)
    monthly_scoring = None
    if monthly:
        from re24_monthly import league_month_environment

        lookup, monthly_scoring = league_month_environment(pitches, rows.season.unique())
        rows = rows.merge(lookup, on=["game_pk", "inning_topbot"], how="left", validate="many_to_one")
        if rows.model_team_rpg.isna().any():
            raise ValueError("No league-month scoring environment for a modeled PA")
        # The existing design expects combined RPG and converts to team RPG.
        rows["runs_per_game"] = 2 * rows.pop("model_team_rpg")
    if rounded:
        from re24_rounded_inputs import rounded_inputs

        rows = rounded_inputs(rows, pitches, project / "data/reference/fangraphs_woba_reliability.csv")
    return rows, monthly_scoring, pitches


def validate(valid, predictions):
    """Return 2024 metrics, per-state calibration, and paired bootstrap intervals."""
    evaluation = pd.DataFrame(metrics(predictions, valid),
                              columns=["model", "n_2024", "rmse_2024", "negative_predictions"])
    calibration, intervals = [], []
    rng = np.random.default_rng(2024)
    for name, prediction in predictions.items():
        scored = valid[["game_pk", "outs", "bases", "runs_remaining"]].assign(prediction=prediction)
        cells = scored.groupby(["outs", "bases"]).agg(
            n=("runs_remaining", "size"), observed=("runs_remaining", "mean"),
            predicted=("prediction", "mean"),
        ).reset_index()
        assert len(cells) == 24 and cells.n.sum() == len(valid)
        calibration.append(cells.assign(model=name))
    for challenger, baseline in (
        ("positive_woba", "constant_woba_slope"),
        ("positive_woba", "unconstrained"),
        ("unconstrained", "constant_woba_slope"),
    ):
        inputs = valid.assign(pred_m1=predictions[baseline], pred_m2=predictions[challenger])
        intervals.append({"comparison": f"{challenger} minus {baseline}", **bootstrap(inputs, rng)})
    return evaluation, pd.concat(calibration, ignore_index=True), pd.DataFrame(intervals)


def leave_one_season_out(rows, predictions_2024):
    folds = []
    for held_year in sorted(rows.season.unique()):
        fold_train = rows[rows.season.ne(held_year)]
        fold_valid = rows[rows.season.eq(held_year)]
        fold_predictions = predictions_2024 if held_year == 2024 else fit_models(fold_train, fold_valid)[0]
        held_min = fold_valid.runs_per_game.min() / 2
        held_max = fold_valid.runs_per_game.max() / 2
        train_min = fold_train.runs_per_game.min() / 2
        train_max = fold_train.runs_per_game.max() / 2
        folds.extend((held_year, held_min, held_max, train_min, train_max, *metric)
                     for metric in metrics(fold_predictions, fold_valid))
    return pd.DataFrame(folds, columns=[
        "held_out_year", "held_min_rpg", "held_max_rpg", "train_min_rpg", "train_max_rpg",
        "model", "n", "rmse", "negative_predictions",
    ])


def effect_tables(fit, rpg_reference):
    """Return wOBA slopes by state and runner speed slopes by base over the RPG grid."""
    low, high = fit.x[WOBA_LOW], fit.x[WOBA_HIGH]
    assert min(low.min(), high.min()) >= MIN_WOBA_EFFECT - 1e-12
    effects = pd.DataFrame({"outs": np.arange(24) // 8, "bases": np.arange(24) % 8})
    rpg_grid = np.linspace(RPG_LOW, RPG_HIGH, round((RPG_HIGH - RPG_LOW) / RPG_UNIT) + 1)
    for rpg in rpg_grid:
        weight = (rpg - RPG_LOW) / (RPG_HIGH - RPG_LOW)
        effects[f"woba_gain_at_{rpg:.2f}"] = (1 - weight) * low + weight * high
    runner = pd.DataFrame({"base": [1, 2, 3]})
    for rpg in rpg_grid:
        runner[f"runs_per_fts_at_{rpg:.2f}"] = (
            fit.x[RUNNER] + (rpg - rpg_reference) / RPG_UNIT * fit.x[RUNNER_RPG]
        )
    return effects, runner


def main(monthly=False, rounded=False, pitch_dir=None, first_season=FIRST_SEASON):
    project = Path(__file__).resolve().parents[1]
    if pitch_dir is None:
        pitch_dir = project / "../../07_Datasets/01_Statcast/raw/pitch_by_pitch"
    rows, monthly_scoring, _ = load_rows(project, pitch_dir, monthly, rounded, first_season)
    train = rows[rows.season.between(first_season, VALID_SEASON - 1)]
    valid = rows[rows.season.eq(VALID_SEASON)]
    assert set(train.season) == set(range(first_season, VALID_SEASON))
    assert len(valid) and train.game_date.max() < valid.game_date.min()

    predictions, fit, unconstrained, rpg_reference, speed_reference = fit_models(train, valid)
    evaluation, calibration, intervals = validate(valid, predictions)
    loso = leave_one_season_out(rows, predictions)
    effects, runner = effect_tables(fit, rpg_reference)
    bound_hits = np.count_nonzero(fit.active_mask[WOBA_LOW]) + np.count_nonzero(fit.active_mask[WOBA_HIGH])
    negative_endpoints = np.count_nonzero(unconstrained[WOBA_LOW] < 0) + np.count_nonzero(unconstrained[WOBA_HIGH] < 0)
    seasons = rows.groupby("season").agg(
        n=("season", "size"), mean_team_rpg=("runs_per_game", lambda x: x.mean() / 2),
        rpg_levels=("runs_per_game", "nunique"),
    )

    name = "monotone-league-month" if monthly else "monotone-375-500"
    name = name + "-rounded" if rounded else name
    if first_season != FIRST_SEASON:
        name += f"-from{first_season}"
    output = project / "outputs" / name
    output.mkdir(parents=True, exist_ok=True)
    if monthly:
        monthly_scoring.to_csv(output / "league_month_scoring.csv", index=False)
    evaluation.to_csv(output / "evaluation.csv", index=False)
    loso.to_csv(output / "leave_one_season_out.csv", index=False)
    intervals.to_csv(output / "bootstrap.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    effects.to_csv(output / "batter_effects.csv", index=False)
    runner.to_csv(output / "runner_effects.csv", index=False)
    seasons.to_csv(output / "seasons.csv")
    np.savez(output / "fit.npz", beta=fit.x, rpg_reference=rpg_reference,
             speed_reference=speed_reference, train_rpg=train.runs_per_game.unique() / 2,
             rpg_low=RPG_LOW, rpg_high=RPG_HIGH, rpg_unit=RPG_UNIT)

    print("Training team RPG values:", np.sort(train.runs_per_game.unique() / 2))
    print(seasons.to_string())
    print(evaluation.to_string(index=False))
    print(loso.to_string(index=False))
    print(f"Endpoint wOBA slopes at the lower bound: {bound_hits}/48")
    print(f"Negative unconstrained endpoint slopes: {negative_endpoints}/48")
    print(effects.query("outs == 1 and bases == 2").to_string(index=False))
    print(runner.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly", action="store_true", help="Use AL/NL league-month scoring.")
    parser.add_argument("--rounded", action="store_true", help="Use FanGraphs-style wOBA and 1 ft/s speed.")
    parser.add_argument("--pitches", type=Path, help="Root of the monthly Statcast Parquet archive.")
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON, choices=range(2021, VALID_SEASON),
                        help="First training season. 2024 is always the validation season.")
    args = parser.parse_args()
    main(monthly=args.monthly, rounded=args.rounded, pitch_dir=args.pitches, first_season=args.first_season)
