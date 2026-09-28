from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_data"
ODDS_DIR = ROOT / "odds"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

# adapter à tes saisons réelles
SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]

FILES = {
    "scores":    "nhl_{season}_scores.parquet",
    "team_stats":"nhl_{season}_team_stats.parquet",
    "on_ice":    "nhl_{season}_on_ice_stats.parquet",
    "shifts":    "nhl_{season}_shifts.parquet",
    "pbp":       "nhl_{season}_pbp.parquet",
}

STRENGTH_5V5 = "5v5"