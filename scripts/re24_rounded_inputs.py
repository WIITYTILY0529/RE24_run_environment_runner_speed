"""Make retrospective player-season wOBA and integer Sprint Speed inputs."""

import numpy as np
import pandas as pd

from dataset import PA, POSITIVE_EVENTS, SEASON_WEIGHTS, WOBA_WEIGHTS


def season_woba(pitches, reliability_path):
    """Return regressed wOBA by batter-season and the value used for a batter with no PA.

    `pitches` is sorted dataset.read_pitches() output.
    """
    pitches = pitches[pitches.season.isin(SEASON_WEIGHTS)]
    ends = pitches.drop_duplicates(PA, keep="last")[["season", "batter", "events"]].copy()
    ends["value"] = ends.events.map(WOBA_WEIGHTS)
    for year, weights in SEASON_WEIGHTS.items():
        selected = ends.season.eq(year) & ends.events.isin(POSITIVE_EVENTS)
        ends.loc[selected, "value"] = ends.loc[selected, "events"].map(dict(zip(POSITIVE_EVENTS, weights)))
    hitters = (ends.dropna(subset=["value"])
               .groupby(["season", "batter"]).value.agg(raw_woba="mean", opportunities="size"))

    # FanGraphs publishes reliability and summed sample means at 10-PA intervals.
    curve = pd.read_csv(reliability_path).sort_values("denom")
    counts = np.r_[0, curve.denom]
    alpha = np.r_[0, curve.alpha]
    means = curve.sample_mean / curve.denom
    prior = np.r_[means.iloc[0], means]
    hitters["woba_alpha"] = np.interp(hitters.opportunities, counts, alpha)
    mean = np.interp(hitters.opportunities, counts, prior)
    hitters["fg_style_woba"] = hitters.woba_alpha * hitters.raw_woba + (1 - hitters.woba_alpha) * mean
    return hitters, prior[0]


def rounded_inputs(rows, pitches, reliability_path):
    """Add rounded wOBA and speeds. `pitches` is sorted dataset.read_pitches() output."""
    hitters, no_pa_woba = season_woba(pitches, reliability_path)
    result = rows.merge(hitters[["fg_style_woba"]], on=["season", "batter"],
                        how="left", validate="many_to_one")
    result["woba"] = np.floor(result.pop("fg_style_woba").fillna(no_pa_woba) * 100 + .5) / 100
    speeds = ["speed_1", "speed_2", "speed_3"]
    result[speeds] = np.floor(result[speeds] + .5)
    assert len(result) == len(rows) and result.woba.notna().all()
    assert result[speeds].isna().equals(rows[speeds].reset_index(drop=True).isna())
    return result
