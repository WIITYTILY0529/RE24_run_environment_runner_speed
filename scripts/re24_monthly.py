"""Build retrospective AL/NL month scoring environments from Statcast."""

import numpy as np
import pandas as pd

AL = set("ATH OAK BAL BOS CLE CWS DET HOU KC LAA MIN NYY SEA TB TEX TOR".split())
NL = set("ATL AZ CHC CIN COL LAD MIA MIL NYM PHI PIT SD SF STL WSH".split())
LEAGUE = {**dict.fromkeys(AL, "AL"), **dict.fromkeys(NL, "NL")}


def league_month_environment(pitches, seasons):
    """Return team-half RPG lookup and summary from dataset.read_pitches() output."""
    pitches = pitches[pitches.season.isin(seasons)]
    games = pitches.groupby("game_pk", as_index=False).agg(
        date=("game_date", "min"), home_team=("home_team", "first"),
        away_team=("away_team", "first"), home_score=("home_score", "max"),
        away_score=("away_score", "max"), post_home_score=("post_home_score", "max"),
        post_away_score=("post_away_score", "max"),
    )
    date = pd.to_datetime(games.date)
    games["season"] = date.dt.year
    games["period"] = date.dt.month.replace({3: 4, 10: 9})
    clubs = []
    for side, half in (("home", "Bot"), ("away", "Top")):
        club = games[["game_pk", "season", "period"]].assign(
            league=games[f"{side}_team"].map(LEAGUE), inning_topbot=half,
            runs=games[[f"{side}_score", f"post_{side}_score"]].max(axis=1),
        )
        clubs.append(club)
    clubs = pd.concat(clubs, ignore_index=True)
    if clubs[["league", "runs"]].isna().any().any():
        raise ValueError("Missing league or score on a regular-season scoreboard")
    keys = ["season", "period", "league"]
    summary = clubs.groupby(keys).runs.agg(team_games="size", runs="sum", team_rpg="mean").reset_index()
    np.testing.assert_equal(int(summary.team_games.sum()), 2 * len(games))
    np.testing.assert_allclose(summary.team_rpg, summary.runs / summary.team_games)
    summary["model_team_rpg"] = np.floor(4 * summary.team_rpg + .5) / 4
    lookup = clubs.merge(summary[keys + ["model_team_rpg"]], on=keys, validate="many_to_one")
    return lookup[["game_pk", "inning_topbot", "model_team_rpg"]], summary
