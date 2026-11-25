"""Fit RE24 with wOBA and runner speed effects that vary with team RPG."""

from pathlib import Path

import numpy as np
import pandas as pd

from dataset import build_dataset


def design(rows, rpg_reference, speed_reference):
    """Build 103 columns with RPG in runs per team per game.

    Columns 48:72 contain centered team RPG by state. Column 102 is
    centered team RPG times the sum of centered measured runner speeds.
    Its single coefficient modifies the speed slopes for all three bases.
    """
    state = np.eye(24)[(8 * rows.outs + rows.bases).to_numpy(dtype=int)]
    rpg = rows.runs_per_game.to_numpy() / 2 - rpg_reference
    woba = 100 * rows.woba.to_numpy()
    speed = rows[["speed_1", "speed_2", "speed_3"]].to_numpy(dtype=float)
    occupied = rows[["runner_1", "runner_2", "runner_3"]].notna().to_numpy()
    missing = occupied & np.isnan(speed)
    runner = np.where(occupied & ~missing, speed - speed_reference, 0)
    return np.column_stack((
        state,
        state * woba[:, None],
        state * rpg[:, None],
        runner,
        missing,
        state * (woba * rpg)[:, None],
        rpg * runner.sum(axis=1),
    ))


def main():
    project = Path(__file__).resolve().parents[1]
    rows, _ = build_dataset(
        project / "../../07_Datasets/01_Statcast/raw/pitch_by_pitch",
        project / "data/raw/sprint_speed",
    )
    train = rows[rows.season.isin([2022, 2023])]
    rpg_reference = train.runs_per_game.mean() / 2
    speed_reference = np.nanmean(train[["speed_1", "speed_2", "speed_3"]].to_numpy())

    x = design(train, rpg_reference, speed_reference)
    y = train.runs_remaining.to_numpy()
    additive = np.linalg.lstsq(x[:, :78], y, rcond=None)[0]
    interaction = np.linalg.lstsq(x, y, rcond=None)[0]
    del x

    evaluation = []
    for year in (2024, 2025, 2026):
        test = rows[rows.season.eq(year)]
        x = design(test, rpg_reference, speed_reference)
        y = test.runs_remaining.to_numpy()
        additive_rmse = np.sqrt(np.mean((x[:, :78] @ additive - y) ** 2))
        interaction_rmse = np.sqrt(np.mean((x @ interaction - y) ** 2))
        evaluation.append((year, len(test), additive_rmse, interaction_rmse,
                           interaction_rmse - additive_rmse))
    evaluation = pd.DataFrame(evaluation, columns=[
        "season", "n", "additive_rmse", "interaction_rmse", "rmse_change"
    ])

    # wOBA slopes vary by base-out state. The runner interaction is shared.
    effects = []
    for rpg in (4.50, 5.00):
        delta_rpg = rpg - rpg_reference
        effects.extend(
            (rpg, state // 8, state % 8, interaction[24 + state] + delta_rpg * interaction[78 + state])
            for state in range(24)
        )
    batter = pd.DataFrame(effects, columns=["team_rpg", "outs", "bases", "runs_per_010_woba"])
    runner = pd.DataFrame({
        "base": [1, 2, 3],
        "runs_per_1_fts_at_4_50": interaction[72:75] + (4.50 - rpg_reference) * interaction[-1],
        "runs_per_1_fts_at_5_00": interaction[72:75] + (5.00 - rpg_reference) * interaction[-1],
    })

    output = project / "outputs/interactions"
    output.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(output / "evaluation.csv", index=False)
    batter.to_csv(output / "batter_effects.csv", index=False)
    runner.to_csv(output / "runner_effects.csv", index=False)
    print("Training team RPG:", sorted(train.runs_per_game.unique() / 2))
    print(evaluation.to_string(index=False, float_format=lambda value: f"{value:.8f}"))
    print("\nOne out, runner on second. Expected runs per .010 wOBA:")
    print(batter[(batter.outs == 1) & (batter.bases == 2)].to_string(index=False))
    print("\nExpected runs per additional ft/s, by occupied base:")
    print(runner.to_string(index=False))


if __name__ == "__main__":
    main()
