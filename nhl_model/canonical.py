# nhl_model/canonical.py
import pandas as pd


def normalize_team_names(scores: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    if "teamId_abbrev" not in shifts.columns or "team_name" not in shifts.columns:
        return scores
    mapping = (shifts[["team_name", "teamId_abbrev"]]
               .dropna().drop_duplicates()
               .set_index("team_name")["teamId_abbrev"].to_dict())
    s = scores.copy()
    s["home_team"] = s["home_team"].map(mapping).fillna(s["home_team"])
    s["away_team"] = s["away_team"].map(mapping).fillna(s["away_team"])
    return s


def goals_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Buts 60 min et OT par game_id, à partir des events GOAL."""
    p = pbp.copy()
    p["game_id"] = p["game_id"].astype(str)
    p["period"] = pd.to_numeric(p["period"], errors="coerce").fillna(0).astype(int)
    p["isHome"] = pd.to_numeric(p["isHome"], errors="coerce").fillna(0).astype(int)

    # buts uniquement (exclut les tirs de barrage)
    g = p[(p["Event"].astype(str).str.upper() == "GOAL")
          & (p["periodType"].astype(str).str.upper() != "SO")
          & (p["period"] <= 4)].copy()

    g60 = g[g["period"] <= 3]
    got = g[g["period"] == 4]

    agg60 = (g60.groupby("game_id")
                .apply(lambda x: pd.Series({
                    "home_goals_60": (x["isHome"] == 1).sum(),
                    "away_goals_60": (x["isHome"] == 0).sum(),
                }), include_groups=False)
                .reset_index())

    aggot = (got.groupby("game_id")
                .apply(lambda x: pd.Series({
                    "home_goals_ot": (x["isHome"] == 1).sum(),
                    "away_goals_ot": (x["isHome"] == 0).sum(),
                }), include_groups=False)
                .reset_index())

    out = agg60.merge(aggot, on="game_id", how="outer").fillna(0)
    for c in ["home_goals_60","away_goals_60","home_goals_ot","away_goals_ot"]:
        out[c] = out[c].astype(int)
    return out


def build_matches(scores, pbp, shifts=None, only_regular=True):
    m = scores.copy()
    m["game_id"] = m["game_id"].astype(str)
    m["season"] = m["season"].astype(str)

    if shifts is not None:
        m = normalize_team_names(m, shifts)

    if only_regular:
        m = m[m["game_id"].str[4:6] == "02"].reset_index(drop=True)

    # dans build_matches, après le filtrage only_regular
    m["home_team"] = m["home_team"].replace({"ARI": "UTA", "PHX": "UTA"})
    m["away_team"] = m["away_team"].replace({"ARI": "UTA", "PHX": "UTA"})

    goals = goals_from_pbp(pbp)
    m = m.merge(goals, on="game_id", how="left")

    for c in ["home_goals_60","away_goals_60","home_goals_ot","away_goals_ot"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0).astype(int)

    # fallback : si pbp manquant pour ce match, déduire du score final
    # (REG : tout en 60 min ; OT/SO : on met tout dans 60 min, on corrigera)
    mask = m["home_goals_60"] + m["away_goals_60"] + \
           m["home_goals_ot"] + m["away_goals_ot"] == 0
    m.loc[mask, "home_goals_60"] = m.loc[mask, "home_score"]
    m.loc[mask, "away_goals_60"] = m.loc[mask, "away_score"]

    m["_check_home"] = m["home_score"] == m["home_goals_60"] + m["home_goals_ot"]
    m["_check_away"] = m["away_score"] == m["away_goals_60"] + m["away_goals_ot"]

    m["home_win"] = (m["result"] == "HOME_WIN").astype(int)
    m["finish_type"] = m["periods"].map({3:"REG", 4:"OT"}).fillna("SO")

    return m[[
        "game_id","season","home_team","away_team",
        "home_score","away_score",
        "home_goals_60","away_goals_60",
        "home_goals_ot","away_goals_ot",
        "home_win","finish_type","periods",
        "_check_home","_check_away",
    ]]