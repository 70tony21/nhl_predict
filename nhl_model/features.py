# nhl_model/features.py
import numpy as np
import pandas as pd
from .config import STRENGTH_5V5


def team_5v5_rolling(team_stats, window=10, min_periods=3):
    ts = team_stats[team_stats["strength"] == STRENGTH_5V5].copy()
    ts["game_id"] = ts["game_id"].astype(str)

    ts["CF_pct"] = ts["CF"] / (ts["CF"] + ts["CA"]).replace(0, np.nan)
    ts["SF_pct"] = ts["SF"] / (ts["SF"] + ts["SA"]).replace(0, np.nan)
    ts["PDO"] = (
        ts["GF"] / ts["SF"].replace(0, np.nan)
        + 1 - ts["GA"] / ts["SA"].replace(0, np.nan)
    )

    ts = ts.sort_values(["team", "game_id"])
    cols = ["CF_per60","CA_per60","CF_pct","SF_pct","PDO","GF_per60","GA_per60"]

    for c in cols:
        ts[f"{c}_r{window}"] = (
            ts.groupby("team")[c]
              .transform(lambda s: s.shift(1)
                          .rolling(window, min_periods=min_periods).mean())
        )
    return ts


def merge_team_features(matches, team_roll, window=10):
    matches = matches.copy()
    team_roll = team_roll.copy()
    matches["game_id"] = matches["game_id"].astype(str)
    team_roll["game_id"] = team_roll["game_id"].astype(str)

    suffix = f"_r{window}"
    feat_cols = [c for c in team_roll.columns if c.endswith(suffix)]

    base = team_roll[["game_id","team"] + feat_cols].copy()

    home = base.rename(columns={"team":"home_team"})
    home = home.rename(columns={c: f"home_{c}" for c in feat_cols})

    away = base.rename(columns={"team":"away_team"})
    away = away.rename(columns={c: f"away_{c}" for c in feat_cols})

    return (matches
            .merge(home, on=["game_id","home_team"], how="left")
            .merge(away, on=["game_id","away_team"], how="left"))