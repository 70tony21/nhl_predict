# scripts/check_dataset.py
import pandas as pd

ds = pd.read_parquet("../artifacts/dataset.parquet")

print("shape:", ds.shape)
print("saisons:", ds["season"].value_counts().sort_index().to_dict())

print("\n--- buts 60 min ---")
print(ds[["home_goals_60", "away_goals_60"]].mean())

print("\n--- cohérence score final ---")
print("check home :", ds["_check_home"].mean())
print("check away :", ds["_check_away"].mean())

print("\n--- gardiens ---")
print("home_goalie NaN :", ds["home_goalie"].isna().mean())
print("away_goalie NaN :", ds["away_goalie"].isna().mean())

print("\n--- finish_type ---")
print(ds["finish_type"].value_counts())

print("\n--- rolling features NaN rate ---")
feat_cols = [c for c in ds.columns if c.endswith("_r10")]
print(ds[feat_cols].isna().mean().sort_values(ascending=False).head(5))