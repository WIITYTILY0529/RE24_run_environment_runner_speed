"""Compare a re24_monotone.py output folder with a saved reference copy."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-9, 1e-12


def compare_fit(outputs, reference):
    new, old = np.load(outputs / "fit.npz"), np.load(reference / "fit.npz")
    assert set(new.files) == set(old.files), (new.files, old.files)
    for key in old.files:
        np.testing.assert_allclose(new[key], old[key], rtol=RTOL, atol=ATOL, err_msg=key)


def compare_csv(path, reference_path):
    new, old = pd.read_csv(path), pd.read_csv(reference_path)
    assert list(new.columns) == list(old.columns), path.name
    assert len(new) == len(old), path.name
    for column in old.columns:
        if pd.api.types.is_numeric_dtype(old[column]):
            np.testing.assert_allclose(new[column], old[column], rtol=RTOL, atol=ATOL,
                                       err_msg=f"{path.name}:{column}")
        else:
            assert new[column].equals(old[column]), f"{path.name}:{column}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Folder holding the reference outputs.")
    parser.add_argument("--outputs", type=Path, default=Path("outputs/monotone-league-month-rounded"))
    args = parser.parse_args()
    compare_fit(args.outputs, args.reference)
    checked = ["fit.npz"]
    for reference_path in sorted(args.reference.glob("*.csv")):
        path = args.outputs / reference_path.name
        if not path.exists():
            print(f"Skipped {reference_path.name}: not in {args.outputs}")
            continue
        compare_csv(path, reference_path)
        checked.append(reference_path.name)
    print("Matched:", ", ".join(checked))


if __name__ == "__main__":
    main()
