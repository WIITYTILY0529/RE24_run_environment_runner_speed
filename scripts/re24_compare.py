"""Fit the 75-term RE24 model (one wOBA slope per state) and compare it with other models.

Trains on 2022–24 and validates on 2025 by default. Uses league-month team RPG,
full-season FanGraphs-style regressed wOBA, and 1 ft/s Sprint Speed. A runner
with unknown Sprint Speed is treated as an average runner.
"""

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.optimize import lsq_linear

from re24_monotone import MIN_WOBA_EFFECT, RPG_HIGH, RPG_LOW, RPG_UNIT, bootstrap, load_rows, metrics
from re24_rounded_inputs import season_woba

FIRST_SEASON, VALID_SEASON = 2022, 2025
SPEEDS = ["speed_1", "speed_2", "speed_3"]
RUNNERS = ["runner_1", "runner_2", "runner_3"]

# Column blocks of full_design(). Each model uses a subset of them.
STATE = list(range(0, 24))        # base-out state intercepts (state = 8 * outs + bases)
WOBA = list(range(24, 48))        # state x wOBA in .010 units
RPG = list(range(48, 72))         # state x (team RPG - reference) / 0.25
RUNNER = [72, 73, 74]             # centered Sprint Speed on first, second, third
RUNNER_RPG = [75]                 # (team RPG - reference) / 0.25 x summed centered speed
WOBA_LOW = list(range(76, 100))   # state x wOBA weighted toward RPG_LOW
WOBA_HIGH = list(range(100, 124)) # state x wOBA weighted toward RPG_HIGH
N_COLUMNS = 124

FINAL = "final_75"
# name: (columns, columns bounded below by MIN_WOBA_EFFECT)
# Each comparison model differs from the final model in one respect.
MODELS = {
    FINAL: (STATE + WOBA + RPG + RUNNER, WOBA),
    "final_75_unconstrained": (STATE + WOBA + RPG + RUNNER, []),
    "with_speed_rpg_76": (STATE + WOBA + RPG + RUNNER + RUNNER_RPG, WOBA),
    "woba_rpg_interaction_99": (STATE + WOBA_LOW + WOBA_HIGH + RPG + RUNNER, WOBA_LOW + WOBA_HIGH),
    "fangraphs_72": (STATE + WOBA + RPG, WOBA),
    "re24_table_24": (STATE, []),
    # Differs in two respects; kept to show both RPG interactions together.
    "both_interactions_100": (STATE + WOBA_LOW + WOBA_HIGH + RPG + RUNNER + RUNNER_RPG, WOBA_LOW + WOBA_HIGH),
}
# Models offered in the HTML explorer, with their labels.
EXPLORER = {
    FINAL: "Final: one wOBA slope per state",
    "woba_rpg_interaction_99": "wOBA slope changes with RPG",
    "both_interactions_100": "wOBA and speed slopes change with RPG",
}
EXPLORER_HTML = "re24-model-explorer.html"


def full_design(rows, rpg_reference, speed_reference):
    state = np.eye(24)[(8 * rows.outs + rows.bases).to_numpy(dtype=int)]
    team_rpg = rows.runs_per_game.to_numpy() / 2
    rpg = (team_rpg - rpg_reference) / RPG_UNIT
    woba = 100 * rows.woba.to_numpy()
    speed = rows[SPEEDS].to_numpy(dtype=float)
    occupied = rows[RUNNERS].notna().to_numpy()
    # Unknown speed -> average runner: its centered speed is 0, like an empty base.
    runner = np.where(occupied & ~np.isnan(speed), speed - speed_reference, 0)
    high_weight = (team_rpg - RPG_LOW) / (RPG_HIGH - RPG_LOW)
    if not np.all((high_weight >= 0) & (high_weight <= 1)):
        raise ValueError("Team RPG is outside the constrained range")
    state_woba = state * woba[:, None]
    x = np.column_stack((
        state, state_woba, state * rpg[:, None], runner, rpg * runner.sum(axis=1),
        state_woba * (1 - high_weight[:, None]), state_woba * high_weight[:, None],
    ))
    if x.shape[1] != N_COLUMNS or not np.isfinite(x).all():
        raise ValueError("Unexpected or non-finite design matrix")
    return x


def fit_models(train, valid):
    """Fit every model on train and return validation predictions and fits."""
    rpg_reference = train.runs_per_game.mean() / 2
    speed_reference = np.nanmean(train[SPEEDS].to_numpy())
    # QR of [X, y] keeps every column subset's least-squares objective exactly.
    triangular = np.empty((0, N_COLUMNS + 1))
    for start in range(0, len(train), 50000):
        batch = train.iloc[start:start + 50000]
        y = batch.runs_remaining.to_numpy()
        if not np.isfinite(y).all():
            raise ValueError("Non-finite runs remaining")
        augmented = np.column_stack((full_design(batch, rpg_reference, speed_reference), y))
        triangular = qr(np.vstack((triangular, augmented)), mode="r")[0][:N_COLUMNS + 1]
    reduced, target = triangular[:, :-1], triangular[:, -1]
    x = full_design(valid, rpg_reference, speed_reference)
    fits, predictions = {}, {}
    for name, (columns, bounded) in MODELS.items():
        lower = np.where(np.isin(columns, bounded), MIN_WOBA_EFFECT, -np.inf)
        fit = lsq_linear(reduced[:, columns], target, bounds=(lower, np.inf), method="bvls")
        if not fit.success:
            raise RuntimeError(f"{name}: {fit.message}")
        fits[name] = fit
        predictions[name] = x[:, columns] @ fit.x
    return predictions, fits, rpg_reference, speed_reference


def validate(valid, predictions):
    evaluation = pd.DataFrame(metrics(predictions, valid), columns=["model", "n", "rmse", "negative_predictions"])
    evaluation.insert(1, "coefficients", [len(MODELS[name][0]) for name in evaluation.model])
    calibration = []
    for name, prediction in predictions.items():
        scored = valid[["outs", "bases", "runs_remaining"]].assign(prediction=prediction)
        cells = scored.groupby(["outs", "bases"]).agg(
            n=("runs_remaining", "size"), observed=("runs_remaining", "mean"),
            predicted=("prediction", "mean"),
        ).reset_index()
        assert len(cells) == 24 and cells.n.sum() == len(valid)
        calibration.append(cells.assign(model=name))
    calibration = pd.concat(calibration, ignore_index=True)
    worst = calibration.assign(error=(calibration.predicted - calibration.observed).abs())
    evaluation["max_state_error"] = evaluation.model.map(worst.groupby("model").error.max())
    rng = np.random.default_rng(2024)
    intervals = []
    for name in predictions:
        if name != FINAL:
            inputs = valid.assign(pred_m1=predictions[name], pred_m2=predictions[FINAL])
            intervals.append({"comparison": f"{FINAL} minus {name}", **bootstrap(inputs, rng)})
    return evaluation, calibration, pd.DataFrame(intervals)


def leave_one_season_out(rows):
    folds = []
    for held_year in sorted(rows.season.unique()):
        fold_valid = rows[rows.season.eq(held_year)]
        predictions = fit_models(rows[rows.season.ne(held_year)], fold_valid)[0]
        folds.extend((held_year, *metric) for metric in metrics(predictions, fold_valid))
    return pd.DataFrame(folds, columns=["held_out_year", "model", "n", "rmse", "negative_predictions"])


def explorer_data(fits, evaluation, calibration, train, valid_season, rpg_reference, speed_reference):
    """Return the explorer models in one form: the wOBA slope is a line between
    its values at RPG_LOW and RPG_HIGH, and the speed slope may change with RPG."""
    rmse = evaluation.set_index("model").rmse
    models = []
    for name, label in EXPLORER.items():
        columns = MODELS[name][0]
        beta = dict(zip(columns, fits[name].x))
        block = lambda cols: [beta[c] for c in cols]
        low, high = (block(WOBA), block(WOBA)) if WOBA[0] in beta else (block(WOBA_LOW), block(WOBA_HIGH))
        models.append({"id": name, "label": label, "terms": len(columns), "rmse": float(rmse[name]),
                       "intercept": block(STATE), "woba_low": low, "woba_high": high, "rpg": block(RPG),
                       "speed": block(RUNNER), "speed_rpg": beta.get(RUNNER_RPG[0], 0.0)})
    observed = calibration[calibration.model.eq(FINAL)].sort_values(["outs", "bases"])
    return {"models": models, "rpg_reference": float(rpg_reference), "speed_reference": float(speed_reference),
            "rpg_low": RPG_LOW, "rpg_high": RPG_HIGH, "rpg_unit": RPG_UNIT,
            "train_rpg": sorted(float(v) for v in train.runs_per_game.unique() / 2),
            "train_seasons": [int(train.season.min()), int(train.season.max())], "valid_season": valid_season,
            "observed": [[float(m), int(n)] for m, n in zip(observed.observed, observed.n)]}


def term_names():
    states = [f"{outs}out_bases{bases}" for outs in range(3) for bases in range(8)]
    return ([f"intercept_{s}" for s in states] + [f"woba_{s}" for s in states]
            + [f"rpg_{s}" for s in states] + ["speed_1b", "speed_2b", "speed_3b"])


def batter_table(pitches, project, seasons):
    hitters, _ = season_woba(pitches, project / "data/reference/fangraphs_woba_reliability.csv")
    table = hitters.reset_index().rename(columns={"batter": "mlb_player_id"})
    table = table[table.season.isin(seasons)].copy()
    table["mlb_player_id"] = table.mlb_player_id.astype("int64")
    table["model_woba"] = np.floor(table.fg_style_woba * 100 + .5) / 100
    names = pd.read_csv(project / "data/reference/player_names.csv")
    table = table.merge(names, on="mlb_player_id", how="left", validate="many_to_one")
    if table.player_name.isna().any():
        raise ValueError("Add the missing players to data/reference/player_names.csv")
    return table[["season", "mlb_player_id", "player_name", "raw_woba", "opportunities",
                  "woba_alpha", "fg_style_woba", "model_woba"]]


def main(pitch_dir=None, first_season=FIRST_SEASON, valid_season=VALID_SEASON):
    project = Path(__file__).resolve().parents[1]
    if pitch_dir is None:
        pitch_dir = project / "../../07_Datasets/01_Statcast/raw/pitch_by_pitch"
    rows, monthly_scoring, pitches = load_rows(project, pitch_dir, monthly=True, rounded=True,
                                               first_season=first_season, last_season=valid_season)
    batters = batter_table(pitches, project, range(first_season, valid_season))
    del pitches
    train = rows[rows.season.between(first_season, valid_season - 1)]
    valid = rows[rows.season.eq(valid_season)]
    assert set(train.season) == set(range(first_season, valid_season))
    assert len(valid) and train.game_date.max() < valid.game_date.min()

    predictions, fits, rpg_reference, speed_reference = fit_models(train, valid)
    evaluation, calibration, intervals = validate(valid, predictions)
    loso = leave_one_season_out(rows)
    final = fits[FINAL]
    assert len(final.x) == 75 and final.x[24:48].min() >= MIN_WOBA_EFFECT - 1e-12
    coefficients = pd.DataFrame({"term": term_names(), "value": final.x})
    effects = pd.DataFrame({"outs": np.arange(24) // 8, "bases": np.arange(24) % 8,
                            "runs_per_010_woba": final.x[24:48], "at_lower_bound": final.active_mask[24:48] != 0})
    runner = pd.DataFrame({"base": [1, 2, 3], "runs_per_fts": final.x[72:75]})
    seasons = rows.groupby("season").agg(
        n=("season", "size"), mean_team_rpg=("runs_per_game", lambda x: x.mean() / 2),
        min_team_rpg=("runs_per_game", lambda x: x.min() / 2), max_team_rpg=("runs_per_game", lambda x: x.max() / 2),
    )

    output = project / "outputs" / f"compare-from{first_season}-valid{valid_season}"
    output.mkdir(parents=True, exist_ok=True)
    monthly_scoring.to_csv(output / "league_month_scoring.csv", index=False)
    evaluation.to_csv(output / "evaluation.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    intervals.to_csv(output / "bootstrap.csv", index=False)
    loso.to_csv(output / "leave_one_season_out.csv", index=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    effects.to_csv(output / "batter_effects.csv", index=False)
    runner.to_csv(output / "runner_effects.csv", index=False)
    seasons.to_csv(output / "seasons.csv")
    batters.to_csv(output / "batter_season_woba.csv", index=False, encoding="utf-8")
    np.savez(output / "fit.npz", beta=final.x, rpg_reference=rpg_reference,
             speed_reference=speed_reference, train_rpg=train.runs_per_game.unique() / 2,
             rpg_low=RPG_LOW, rpg_high=RPG_HIGH, rpg_unit=RPG_UNIT)
    explorer = explorer_data(fits, evaluation, calibration, train, valid_season, rpg_reference, speed_reference)
    (output / "explorer_models.json").write_text(json.dumps(explorer, indent=1), encoding="utf-8")
    # Refresh the data block of the HTML explorer, which embeds these models.
    html_path = project / "outputs" / EXPLORER_HTML
    if html_path.exists():
        pattern = r'(<script id="data" type="application/json">)(.*?)(</script>)'
        html = html_path.read_text(encoding="utf-8")
        if len(re.findall(pattern, html, re.S)) != 1:
            raise ValueError(f"{EXPLORER_HTML} needs exactly one data block")
        data = json.dumps(explorer, separators=(",", ":"))
        html_path.write_text(re.sub(pattern, lambda m: m.group(1) + data + m.group(3), html, flags=re.S),
                             encoding="utf-8")

    print(seasons.to_string())
    print(evaluation.to_string(index=False))
    print(intervals.to_string(index=False))
    print(loso.pivot(index="model", columns="held_out_year", values="rmse").to_string())
    print(f"wOBA slopes at the lower bound: {int(effects.at_lower_bound.sum())}/24")
    print(runner.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitches", type=Path, help="Root of the monthly Statcast Parquet archive.")
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--valid-season", type=int, default=VALID_SEASON)
    args = parser.parse_args()
    main(pitch_dir=args.pitches, first_season=args.first_season, valid_season=args.valid_season)
