# Archived code

These files belong to the earlier seasonal-RPG models. They are kept for reference and are not part of the final fit. They no longer run as they are: `build_dataset()` in `scripts/dataset.py` now takes the output of `read_pitches()` instead of a pitch folder.

| File | What it was |
| --- | --- |
| `re24.py` | Empirical, batter-only and batter-plus-runner RE24 with a 2022–23 train, 2024 validation and 2025/26 test split. It wrote `test_predictions.parquet`, `metadata.json` and related tables. The final fit's `bootstrap()` now lives in `scripts/re24_monotone.py`. |
| `re24_simple.py` | Scored one PA with the 78-coefficient pooled M2 model from `re24.py`'s `metadata.json`. |
| `re24_interactions.py` | The original file, including `main()`, which fitted the unconstrained interaction model on 2022–23 and wrote `outputs/interactions/`. `design()` and the column layout still live in `scripts/re24_interactions.py`. |
| `verify_re24.py` | Checked `re24.py` outputs against raw pitches. Formerly `tests/verify_re24.py`. |
