"""Score one plate appearance with the fitted season-level RE24 model."""

import argparse
import json
from pathlib import Path


def expected_runs(model, outs, bases, woba, runs_per_game, speeds=(None, None, None)):
    """Return each term. Base mask uses 1 for first, 2 for second, 4 for third."""
    if not 0 <= outs <= 2 or not 0 <= bases <= 7:
        raise ValueError("Use 0–2 outs and a base mask from 0 to 7.")
    beta = model["coefficients"]["m2"]
    if model["m2_choice"] != "pooled" or len(beta) != 78:
        raise ValueError("This formula requires the fitted 78-coefficient pooled M2 model.")

    state = outs * 8 + bases
    batter = beta[state] + beta[24 + state] * 100 * (woba - .320)
    run_environment = beta[48 + state] * (runs_per_game - model["environment_reference"])

    runner = 0.0
    for base, speed in enumerate(speeds):
        if bases & (1 << base):
            if speed is None:
                runner += beta[75 + base]
            else:
                runner += beta[72 + base] * (speed - model["speed_reference"])

    return {
        "batter": batter,
        "run_environment": run_environment,
        "runner": runner,
        "expected_runs": batter + run_environment + runner,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outs", type=int, choices=range(3))
    parser.add_argument("bases", type=int, choices=range(8), help="Add 1 for first, 2 for second, 4 for third.")
    parser.add_argument("woba", type=float)
    parser.add_argument("runs_per_game", type=float, help="Season runs per game, both teams combined.")
    for base in ("first", "second", "third"):
        parser.add_argument(f"--{base}", type=float, help="Sprint Speed in ft/s; omit if unknown.")
    parser.add_argument("--model", type=Path, default=Path("outputs/season-rpg/2026/metadata.json"))
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    terms = expected_runs(model, args.outs, args.bases, args.woba, args.runs_per_game,
                          (args.first, args.second, args.third))
    for name, value in terms.items():
        print(f"{name:16} {value:+.6f}")
