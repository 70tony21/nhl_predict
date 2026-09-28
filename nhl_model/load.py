# nhl_model/load.py
import pandas as pd
from .config import RAW_DIR, FILES, SEASONS


def _path(season: str, key: str) -> str:
    return str(RAW_DIR / season / FILES[key].format(season=season))


def _harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)
    return df


def load_season(season: str) -> dict:
    return {
        "scores":     _harmonize(pd.read_parquet(_path(season, "scores"))),
        "team_stats": _harmonize(pd.read_parquet(_path(season, "team_stats"))),
        "on_ice":     _harmonize(pd.read_parquet(_path(season, "on_ice"))),
        "shifts":     _harmonize(pd.read_parquet(_path(season, "shifts"))),
        "pbp":        _harmonize(pd.read_parquet(_path(season, "pbp"))),
    }


def load_all(seasons=SEASONS) -> dict:
    acc = {k: [] for k in ["scores", "team_stats", "on_ice", "shifts", "pbp"]}
    for s in seasons:
        d = load_season(s)
        for k in acc:
            d[k]["season"] = s  # déjà str
            acc[k].append(d[k])
    return {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}