# scripts/fit_model.py
import json
import pandas as pd
from nhl_model.model import fit
from nhl_model.config import OUT_DIR

TRAIN = ["20212022", "20222023", "20232024"]
VAL   = ["20242025"]
TEST  = ["20252026"]


def main():
    ds = pd.read_parquet(OUT_DIR / "dataset.parquet")

    # fusion ARI → UTA
    for col in ["home_team", "away_team"]:
        ds[col] = ds[col].replace({"ARI": "UTA", "PHX": "UTA"})

    ds = ds.dropna(subset=["home_goals_60", "away_goals_60"])

    train = ds[ds["season"].isin(TRAIN)]
    print(f"train: {len(train)}")

    model = fit(train)

    with open(OUT_DIR / "dc_params.json", "w") as f:
        json.dump(model.to_dict(), f, indent=2)

    d = model.to_dict()
    print(f"\nmu    = {d['mu']:.4f}")
    print(f"gamma = {d['gamma']:.4f}")
    print(f"rho   = {d['rho']:.4f}")

    print("\nBeta home (features standardisées) :")
    for f, b in zip(d["features"], d["beta_home"]):
        print(f"  {f:25s}  {b:+.4f}")
    print("\nBeta away :")
    for f, b in zip(d["features"], d["beta_away"]):
        print(f"  {f:25s}  {b:+.4f}")

    print("\nTop attaque :")
    for t, v in sorted(d["att"].items(), key=lambda x: -x[1])[:8]:
        print(f"  {t}  {v:+.3f}")
    print("\nTop défense :")
    for t, v in sorted(d["dfn"].items(), key=lambda x: x[1])[:8]:
        print(f"  {t}  {v:+.3f}")


if __name__ == "__main__":
    main()