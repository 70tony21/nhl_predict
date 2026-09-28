# nhl_model/goalies.py
import pandas as pd


def starter_from_shifts(shifts: pd.DataFrame) -> pd.DataFrame:
    sh = shifts.copy()
    sh["game_id"] = sh["game_id"].astype(str)

    team_col = "teamId_abbrev" if "teamId_abbrev" in sh.columns else "team_name"

    sh = sh[sh["positionCode"].astype(str).str.upper().eq("G")]
    sh["duration_seconds"] = pd.to_numeric(sh["duration_seconds"], errors="coerce")
    sh = sh.dropna(subset=["duration_seconds", team_col, "playerId", "game_id"])

    g = (sh.groupby(["game_id", team_col, "playerId"], as_index=False)
           ["duration_seconds"].sum()
           .sort_values("duration_seconds", ascending=False)
           .groupby(["game_id", team_col])
           .head(1)
           .rename(columns={team_col: "team_name", "playerId": "goalie_id"}))
    g["game_id"] = g["game_id"].astype(str)
    g["goalie_id"] = g["goalie_id"].astype(str)
    return g[["game_id", "team_name", "goalie_id"]]


def starter_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Fallback : premier gardien vu sur la glace à chaque PSTR."""
    p = pbp.copy()
    p["game_id"] = p["game_id"].astype(str)
    p["sortOrder"] = pd.to_numeric(p["sortOrder"], errors="coerce").fillna(0)

    first = (p[p["Event"] == "PSTR"]
             .sort_values(["game_id", "sortOrder"])
             .groupby("game_id")
             .head(1))

    home = first[["game_id", "homeTeam", "homeGoalie_on_id"]].rename(
        columns={"homeTeam": "team_name", "homeGoalie_on_id": "goalie_id"})
    away = first[["game_id", "awayTeam", "awayGoalie_on_id"]].rename(
        columns={"awayTeam": "team_name", "awayGoalie_on_id": "goalie_id"})

    out = pd.concat([home, away], ignore_index=True)
    out["game_id"] = out["game_id"].astype(str)
    out["goalie_id"] = out["goalie_id"].astype(str)
    return out.dropna()


def attach_starters(matches, shifts, pbp):
    base = starter_from_shifts(shifts)
    fb = starter_from_pbp(pbp)

    # on garde base en priorité, on complète avec pbp
    starters = (pd.concat([base, fb], ignore_index=True)
                  .drop_duplicates(subset=["game_id","team_name"], keep="first"))

    home = starters.rename(columns={"team_name": "home_team",
                                    "goalie_id": "home_goalie"})
    away = starters.rename(columns={"team_name": "away_team",
                                    "goalie_id": "away_goalie"})

    return (matches
            .merge(home, on=["game_id", "home_team"], how="left")
            .merge(away, on=["game_id", "away_team"], how="left"))