"""Compare empirical, batter-only and batter-plus-runner run expectancy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import build_dataset


def design(rows, speed_reference, environment_reference):
    state = np.eye(24)[rows.outs.to_numpy(dtype=int) * 8 + rows.bases.to_numpy(dtype=int)]
    base = np.column_stack((
        state, state * (rows.woba.to_numpy()[:, None] - .320) * 100,
        state * (rows.runs_per_game.to_numpy()[:, None] - environment_reference),
    ))
    occupied = rows[["runner_1", "runner_2", "runner_3"]].notna().to_numpy()
    speed = rows[["speed_1", "speed_2", "speed_3"]].to_numpy(dtype=float, na_value=np.nan)
    missing = occupied & np.isnan(speed)
    centered = np.where(occupied & ~missing, speed - speed_reference, 0)
    pooled = np.column_stack((base, centered, missing))
    states = np.arange(24)
    interactions = []
    for index in range(3):
        active = states[(states % 8 & (1 << index)) != 0]
        contrast = state[:, active[1:]] - state[:, active[:1]]
        interactions.append(contrast * centered[:, index, None])
    return base, pooled, np.column_stack(interactions)


def coefficients(gram, cross, width, penalty=0):
    system = gram[:width, :width].copy()
    if penalty:
        system[np.arange(78, width), np.arange(78, width)] += penalty
    return np.linalg.lstsq(system, cross[:width], rcond=None)[0]


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def bootstrap(rows, rng, repeats=2000):
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


def run(pitches, speeds, output, test_year):
    output.mkdir(parents=True, exist_ok=True)
    rows, audit = build_dataset(pitches, speeds, end_year=test_year)
    train = rows[rows.season.isin([2022, 2023])]
    validation = rows[rows.season.eq(2024)]
    test = rows[rows.season.eq(test_year)].copy()
    assert all(len(split) for split in (train, validation, test))
    assert train.game_date.max() < validation.game_date.min() < test.game_date.min()
    assert validation.game_date.max() < test.game_date.min()

    matrix = (train.groupby(["outs", "bases"]).runs_remaining
              .agg(n="size", re="mean").reset_index())
    scoring = pd.DataFrame(audit["season_scoring"])
    scoring.to_csv(output / "season_scoring.csv", index=False)
    seasonal_re24 = (
        rows.groupby(["season", "outs", "bases"]).runs_remaining
        .agg(n="size", re="mean").reset_index()
        .merge(scoring[["season", "runs_per_game"]], on="season", validate="many_to_one")
    )
    assert seasonal_re24.groupby("season").size().eq(24).all()
    seasonal_re24.to_csv(output / "season_re24.csv", index=False)
    assert len(matrix) == 24
    matrix.to_csv(output / "train_matrix.csv", index=False)
    speed_columns = ["speed_1", "speed_2", "speed_3"]
    speed_reference = float(np.nanmean(train[speed_columns].to_numpy(dtype=float)))
    if not np.isfinite(speed_reference):
        raise ValueError("The training set requires measured runner speed")
    environment_reference = float(train.runs_per_game.mean())
    pooled_width = 78
    x0, xp, xd = design(train, speed_reference, environment_reference)
    x = np.column_stack((xp, xd))
    gram = x.T @ x
    cross = x.T @ train.runs_remaining.to_numpy()
    del x, x0, xp, xd
    m1 = coefficients(gram, cross, 72)
    _, vp, vd = design(validation, speed_reference, environment_reference)
    vx = np.column_stack((vp, vd))
    candidates = {"pooled": coefficients(gram, cross, pooled_width)}
    for penalty in (100, 1000, 10000, 100000):
        candidates[f"ridge_{penalty}"] = coefficients(gram, cross, gram.shape[0], penalty)
    validation_rmse = {
        name: rmse(vx[:, :len(beta)] @ beta - validation.runs_remaining)
        for name, beta in candidates.items()
    }
    choice = min(validation_rmse, key=validation_rmse.get)
    m2 = candidates[choice]

    tx1, txp, txd = design(test, speed_reference, environment_reference)
    tx = np.column_stack((txp, txd))
    state_index = test.outs.to_numpy(dtype=int) * 8 + test.bases.to_numpy(dtype=int)
    test["pred_m0"] = matrix.re.to_numpy()[state_index]
    test["pred_m1"] = tx1 @ m1
    test["pred_m2"] = tx[:, :len(m2)] @ m2
    test.to_parquet(output / "test_predictions.parquet", index=False)

    evaluation, calibration, intervals = [], [], []
    rng = np.random.default_rng(2025)
    subsets = {"all": test, "runner_on_measured": test[test[speed_columns].notna().any(axis=1)]}
    for subset, sample in subsets.items():
        intervals.append({"subset": subset, **bootstrap(sample, rng)})
        for model in ("m0", "m1", "m2"):
            predicted = sample[f"pred_{model}"]
            errors = predicted - sample.runs_remaining
            evaluation.append(dict(subset=subset, model=model, n=len(sample),
                                   rmse=rmse(errors), mean_error=errors.mean(),
                                   negative=int(predicted.lt(0).sum())))
            cells = (sample.groupby(["outs", "bases"])
                     .agg(n=("runs_remaining", "size"), observed=("runs_remaining", "mean"),
                          predicted=(f"pred_{model}", "mean")).reset_index())
            calibration.append(cells.assign(subset=subset, model=model))
    pd.DataFrame(evaluation).to_csv(output / "evaluation.csv", index=False)
    pd.concat(calibration).to_csv(output / "calibration.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "bootstrap.csv", index=False)

    baseline = matrix[["outs", "bases"]].assign(
        woba=.320, runs_per_game=environment_reference)
    for index in range(3):
        baseline[f"runner_{index+1}"] = np.where(baseline.bases & (1 << index), 1, np.nan)
        baseline[f"speed_{index+1}"] = np.where(baseline.bases & (1 << index), speed_reference, np.nan)

    def predict_scenarios(frame):
        _, pooled, deviations = design(frame, speed_reference, environment_reference)
        return np.column_stack((pooled, deviations))[:, :len(m2)] @ m2

    base_prediction = predict_scenarios(baseline)
    ordinary = []
    for _, group in train.groupby(["outs", "bases"]):
        environment = group.runs_per_game.to_numpy(dtype=float) - environment_reference
        fit = np.linalg.lstsq(
            np.column_stack((np.ones(len(group)), environment)), group.runs_remaining, rcond=None
        )[0]
        ordinary.append(fit[0])
    effects = matrix.assign(ordinary_re=ordinary, adjusted_re=base_prediction)
    effects["difference"] = effects.adjusted_re - effects.ordinary_re
    effects["woba_plus_010"] = predict_scenarios(baseline.assign(woba=.330)) - base_prediction
    effects["woba_supported"] = False
    for (outs, bases), group in train.groupby(["outs", "bases"]):
        row = (effects.outs.eq(outs)) & (effects.bases.eq(bases))
        effects.loc[row, "woba_supported"] = group.woba.min() <= .320 and group.woba.max() >= .330
    for index, col in enumerate(speed_columns):
        on_base = (effects.bases & (1 << index)).ne(0)
        effects[f"{col}_plus_1"] = np.where(
            on_base, predict_scenarios(baseline.assign(**{col: baseline[col] + 1})) - base_prediction,
            np.nan)
        support = train.groupby(["outs", "bases"])[col].agg(["min", "max"]).reset_index()
        effects[f"{col}_supported"] = (on_base & support["min"].le(speed_reference)
                                       & support["max"].ge(speed_reference + 1))
    effects.to_csv(output / "effects.csv", index=False)

    metadata = {
        "dataset": audit, "splits": {name: {"rows": len(frame), "games": int(frame.game_pk.nunique()),
                                          "start": str(frame.game_date.min()), "end": str(frame.game_date.max())}
                                       for name, frame in (("train", train), ("validation", validation), ("test", test))},
        "m2_choice": choice, "validation_rmse": validation_rmse,
        "environment_reference": environment_reference, "speed_reference": speed_reference,
        "coverage_by_base_test": {
            str(i): {"occupied": int(test[f"runner_{i}"].notna().sum()),
                     "measured": int(test[f"speed_{i}"].notna().sum())}
            for i in (1, 2, 3)
        },
        "coefficients": {"m1": m1.tolist(), "m2": m2.tolist()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(pd.DataFrame(evaluation).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitches", type=Path,
                        default=Path("../../07_Datasets/01_Statcast/raw/pitch_by_pitch"))
    parser.add_argument("--speeds", type=Path, default=Path("data/raw/sprint_speed"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--test-year", type=int, choices=(2025, 2026), default=2025)
    args = parser.parse_args()
    run(args.pitches, args.speeds, args.output, args.test_year)
