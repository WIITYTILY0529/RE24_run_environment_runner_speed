# Batter- and runner-adjusted RE24

This project estimates how many runs the batting team will score **from the first pitch of a plate appearance through the end of that half-inning**. The final model adjusts the estimate for batter wOBA, the scoring environment, and the Sprint Speed of each runner already on base.

## Open the final HTML

Open [re24-rounded-explorer.html](outputs/re24-rounded-explorer.html) in a browser. It is a self-contained file that works offline without Python or a server.

1. Click the bases on the diamond to add or remove runners, and select **0**, **1**, or **2** outs.
2. Choose **regressed batter wOBA** in **.010** steps and **team RPG** from **3.75 to 5.00** in **0.25** steps. RPG means runs per **team** per game.
3. Set the Sprint Speed of each occupied runner in **1 ft/s** steps. Only occupied bases show a speed selector. For a runner whose speed is unknown, choose the training average, about 27 ft/s.
4. Compare **Observed 2025 RE24** with the **Model prediction**. The observed value averages eligible 2025 starts in the selected base-out state across batters and scoring environments. The prediction uses the settings you choose.
5. Switch between **Table** and **Graph** to see the contributions to the difference. **What one step adds** shows the predicted change from +.010 wOBA, or from +1 ft/s for one runner, holding the other settings fixed.

The breakdown uses .320 wOBA as a display reference. The model fit uses raw regressed wOBA in .010 units. The **3.75** and **5.00 RPG** options are marked extrapolated because the 2022–24 training data contain four rounded levels from **4.00 to 4.75**.

### Three-model explorer

[re24-model-explorer.html](outputs/re24-model-explorer.html) has the same controls, plus a selector for three models: the final 75-term model, the 99-term model with wOBA × RPG, and the 100-term model with both wOBA × RPG and speed × RPG. A table under the result shows all three models' predictions and step values for the current situation. Change RPG between 4.00 and 4.75 to see the two interaction models change while the final model stays fixed. A link can open a specific model with `#final-75`, `#woba-rpg-99`, or `#both-100`.

`re24_compare.py` refreshes this file's embedded data on every run from `explorer_models.json`, so it always matches the latest fit. The page is also published as a private claude.ai artifact; share it from the page's Share menu.

## Repository layout

- `scripts/` contains the model and preprocessing code.
- `data/raw/` contains the source Sprint Speed files. `data/reference/` contains the reliability, suspension, and player-name tables.
- `outputs/` contains the final HTML and the fits.
- `tests/` contains `check_monotone_outputs.py`, which compares a fresh `re24_monotone.py` fit with a saved copy of its outputs.
- `archive/` contains the code for the earlier seasonal models. See [archive/README.md](archive/README.md).

## Source code

| File | Role |
| --- | --- |
| [`re24_compare.py`](scripts/re24_compare.py) | **Final model.** Fits the 75-term model and five comparison models, validates them on 2025, and exports results, named coefficients, and the batter-season wOBA table. |
| [`dataset.py`](scripts/dataset.py) | `read_pitches()` reads Statcast regular-season pitches once. `build_dataset()` constructs eligible PA starts and runs-remaining targets and joins runner speeds. Also holds the seasonal FanGraphs wOBA weights for 2021–25. |
| [`re24_monthly.py`](scripts/re24_monthly.py) | Computes AL/NL league-month scoring levels from the same pitches. |
| [`re24_rounded_inputs.py`](scripts/re24_rounded_inputs.py) | `season_woba()` builds FanGraphs-style regressed wOBA for each batter-season. `rounded_inputs()` rounds wOBA and Sprint Speed. |
| [`re24_monotone.py`](scripts/re24_monotone.py) | Earlier 103-term model. Its `load_rows()` and `bootstrap()` are reused by `re24_compare.py`. |
| [`re24_interactions.py`](scripts/re24_interactions.py) | Supplies the 103-column design used by `re24_monotone.py`. |

## Data and preprocessing

| Input | Location |
| --- | --- |
| Statcast pitch-by-pitch Parquet | Supply the archive root with `--pitches`, with files organized as `YYYY/MM/statcast_YYYY_MM.parquet`. The final model reads 2021–25. 2021 supplies only as-of history features, which the rounded wOBA replaces. |
| Baseball Savant seasonal Sprint Speed | `data/raw/sprint_speed/YYYY.csv` |
| FanGraphs published wOBA reliability curve | `data/reference/fangraphs_woba_reliability.csv` |
| Games that resumed on a later date | `data/reference/suspended_games.csv` |
| Player names by MLB ID, from the MLB Stats API | `data/reference/player_names.csv` |

1. **Reconstruct plate appearance starts.** Keep regular-season pitches. Sort by game, half-inning, PA, and pitch, then use the first pitch to read the outs, runners, batter, and batting-team score. Define the target as the batting team's verified final half-inning score minus its score at PA start. The next half-inning's fielding score supplies the final score. The PA modeling sample excludes extra innings, halves without a following half to confirm the score, and listed later-resumed games.
2. **Calculate the scoring environment.** Recover each team's final game score and group by its league (AL or NL), season, and month. Combine March with April and September with October. For each group, divide runs by the number of team-game appearances and round the team RPG to the nearest **0.25**. Each PA receives the value for its batting team's league and month. An interleague game can therefore have different RPG inputs for its two clubs. The exported scoring table includes both the original and rounded values.
3. **Build batter wOBA.** Use each batter's full-season PA outcomes and that season's FanGraphs Guts event weights. The wOBA denominator is AB + BB − IBB + SF + HBP. Interpolate the published reliability and sample mean by PA count, as in the FanGraphs method. The regressed input is `reliability × observed wOBA + (1 − reliability) × sample mean`, rounded to the nearest **.010**. Every PA in a season uses the batter's full-season value, even though RPG changes by month.
4. **Build individual runner inputs.** Match MLB player IDs to the same-season Sprint Speed file first, then use the closest available season, preferring the earlier year in a tie. Round measured speeds to the nearest **1 ft/s**. A runner whose speed is still unknown counts as an average runner. About 0.1% of runners on each base are unknown. An empty base contributes zero.

This is a **retrospective** model. Full-season batter wOBA, full-month scoring, and some runner speed measurements can include information available after the PA.

## Final model: 75 terms

The model fits **24 base-out states jointly** by least squares. For state *s*:

```
expected runs = intercept_s
              + wOBA_s × (100 × wOBA)
              + RPG_s × (team RPG − training mean) / 0.25
              + for each occupied base b: speed_b × (Sprint Speed − training mean)
```

| Terms | Count |
| --- | --- |
| State intercepts | 24 |
| wOBA slope per state, per +.010 wOBA | 24 |
| RPG effect per state, per +0.25 team RPG | 24 |
| Speed slope per base, per +1 ft/s | 3 |
| **Total** | **75** |

The wOBA slopes are bounded to at least 0.000001 runs per +.010 wOBA. None reached the bound, so the result equals ordinary least squares. The code reduces the training data with chunked QR and then solves with SciPy's bounded-variable least squares.

The model trains on **515,255 PA starts from 2022–24** and validates on **171,861 starts from 2025**. 2021 is left out because pitchers still batted in NL parks: they took 2.77% of the 2021 wOBA opportunities and 0.13% or less from 2022 on. The 2025 batter wOBA uses the 2025 Guts weights (.691, .722, .882, 1.252, 1.584, 2.037). With those weights, league wOBA is .3129 against FanGraphs' .313.

### How RPG enters the final model

In the final model the wOBA, RPG, and speed terms are added, never multiplied. RPG therefore raises or lowers a state's expected runs, but it does not change what +.010 wOBA or +1 ft/s is worth. The wOBA slope differs by state, from +0.0172 runs per .010 with 0 outs and bases empty to +0.0543 with 2 outs and bases loaded, but each state's slope is the same at every RPG. The FanGraphs tool makes the same assumption.

Example: 1 out, runner on second at the average speed (27.38 ft/s). The final model's coefficients are +0.0261 runs per .010 wOBA and +0.0126 runs per +0.25 RPG.

| `final_75` | RPG 4.00 | RPG 4.75 |
| --- | --- | --- |
| .320 batter | 0.660 | 0.698 |
| .350 batter | 0.738 | 0.776 |
| Gap (.350 − .320) | 0.078 | 0.078 |
| Runner on second, +1 ft/s | +0.0117 | +0.0117 |

Higher RPG adds the same 0.038 runs to both batters, so the gap stays 0.078. Because the gap is a fixed number of runs, it is a slightly smaller share of expected runs in a high-scoring month: 11.8% at 4.00 RPG and 11.2% at 4.75.

The interaction models let those values change with RPG. The models below were refitted on the same 2022–24 data for this example:

| Same example | RPG 4.00 | RPG 4.75 |
| --- | --- | --- |
| `woba_rpg_interaction_99` (wOBA × RPG): gap | 0.052 | 0.096 |
| `woba_rpg_interaction_99`: runner on second, +1 ft/s | +0.0116 | +0.0116 |
| 100-term model (wOBA × RPG and speed × RPG): gap | 0.053 | 0.096 |
| 100-term model: runner on second, +1 ft/s | +0.0052 | +0.0158 |

In these models the batter gap nearly doubles between 4.00 and 4.75 RPG, and in the 100-term model the speed effect triples. The 100-term model had a 2025 RMSE of 0.980085 and the 99-term model 0.980084. Neither did better than the final model's 0.980049. The fitted interactions look large in one state, but over all PAs they move predictions by only a few hundredths of a run, which is too little to show up against per-PA run variance. They are also estimated from only four RPG levels, 4.00 to 4.75.

## Comparison on 2025

All seven models are fitted on the same QR-compressed training data. Each comparison model except `both_interactions_100` differs from the final model in one respect.

| Model | Terms | 2025 RMSE | Final minus model (95% interval) |
| --- | --- | --- | --- |
| `final_75` | 75 | 0.980049 | — |
| `final_75_unconstrained`: no wOBA bound | 75 | 0.980049 | identical |
| `with_speed_rpg_76`: adds a shared speed × RPG term | 76 | 0.980050 | −0.000032 to +0.000029 |
| `woba_rpg_interaction_99`: wOBA slope varies with RPG between the 3.75 and 5.00 endpoints | 99 | 0.980084 | −0.000091 to +0.000020 |
| `fangraphs_72`: no runner terms, as in the FanGraphs tool | 72 | 0.980280 | −0.000360 to −0.000108 |
| `re24_table_24`: state averages only | 24 | 0.981714 | −0.002066 to −0.001243 |
| `both_interactions_100`: wOBA × RPG and speed × RPG together | 100 | 0.980085 | −0.000094 to +0.000023 |

The intervals come from a paired game-level bootstrap with 2,000 resamples. RMSE stays near 0.98 for every model because runs after a single PA vary widely, so compare the intervals rather than the raw RMSE gaps. None of the models produced a negative 2025 prediction.

- **Runner speed helps.** The final model beats `fangraphs_72` and the plain RE24 table, and both intervals exclude 0.
- **Neither RPG interaction helps.** Adding wOBA × RPG (24 terms) or speed × RPG (1 term) gives no measurable improvement, and both intervals include 0. This agrees with the FanGraphs article, which found that wOBA × run-environment interactions did not improve its fit. The interactions are too small to detect with 2022–24 data. That is not proof that they are zero.
- **Speed × RPG behaves badly outside the data.** In `with_speed_rpg_76`, extending the line to 3.75 RPG makes the third-base speed effect negative (−0.0026 runs per ft/s). This is one reason the final model leaves the term out.
- **Leave-one-season-out over 2022–25** puts `re24_table_24` last and `fangraphs_72` second to last in every held-out season. The four models with runner speed are within 0.00005 RMSE of each other, with no consistent winner. `final_75` is best for 2022 and 2025, `woba_rpg_interaction_99` for 2023, and `with_speed_rpg_76` for 2024.

## Reproduce the fit

From the repository root, use Python 3.12 in a virtual environment and install the dependencies. Obtain the monthly pitch-by-pitch Parquet files separately. In the example, their archive root is `data/statcast`. Replace it with the location of your archive.

```sh
python -m pip install -r requirements.txt
python -X utf8 scripts/re24_compare.py --pitches data/statcast
```

The command writes these files to [`outputs/compare-from2022-valid2025`](outputs/compare-from2022-valid2025):
- evaluation, calibration, bootstrap, and leave-one-season-out tables;
- the 75 named coefficients (`coefficients.csv`) and `fit.npz`;
- `batter_season_woba.csv` for 2022–24, with player names;
- `explorer_models.json`, the coefficients of the three explorer models. The same data is written into `outputs/re24-model-explorer.html`.

The final HTML embeds that folder's `fit.npz` and the observed 2025 values from its `calibration.csv`. Running the command does not regenerate the HTML.

## Earlier 103-term fits

`re24_monotone.py` fits the earlier model. It lets the wOBA slope vary with RPG between the 3.75 and 5.00 endpoints, has a missing-speed term per base, and has a shared speed × RPG term. Two fits are kept for comparison:

```sh
python -X utf8 scripts/re24_monotone.py --monthly --rounded --pitches data/statcast
python -X utf8 scripts/re24_monotone.py --monthly --rounded --first-season 2022 --pitches data/statcast
```

- The first trains on 2021–23 and validates on 2024. It writes to [`outputs/monotone-league-month-rounded`](outputs/monotone-league-month-rounded) and had a 2024 RMSE of 0.973004. That folder also holds a 2021–24 `batter_season_woba.csv` with player names; no script writes that file.
- The second trains on 2022–23 and validates on 2024. It writes to [`outputs/monotone-league-month-rounded-from2022`](outputs/monotone-league-month-rounded-from2022) and had a 2024 RMSE of 0.973065.

To confirm a code change leaves these fits unchanged, copy an output folder somewhere first, rerun the fit, and compare:

```sh
python tests/check_monotone_outputs.py path/to/saved-copy
```

## References

- [FanGraphs batter-specific run expectancy](https://blogs.fangraphs.com/introducing-the-batter-specific-run-expectancy-tool/)
- [FanGraphs reliability and regression to the mean](https://blogs.fangraphs.com/a-new-way-to-look-at-sample-size-math-supplement/)
- [FanGraphs seasonal wOBA weights](https://www.fangraphs.com/tools/guts?type=cn)
- [Baseball Savant Sprint Speed](https://baseballsavant.mlb.com/leaderboard/sprint_speed)
