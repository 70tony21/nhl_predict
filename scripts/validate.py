# scripts/validate.py
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from nhl_model.model import PoissonModel, FEATURES
from nhl_model.markets import score_matrix, market_probs
from nhl_model.config import OUT_DIR


def main():
    ds = pd.read_parquet(OUT_DIR / "dataset.parquet")
    for col in ["home_team", "away_team"]:
        ds[col] = ds[col].replace({"ARI": "UTA", "PHX": "UTA"})

    with open(OUT_DIR / "dc_params.json") as f:
        model = PoissonModel.from_dict(json.load(f))

    for season in ["20242025", "20252026"]:
        val = ds[ds["season"] == season].dropna(subset=FEATURES).copy()

        preds = []
        for _, r in val.iterrows():
            X = r[FEATURES].values.astype(float)
            print("teams in model:", sorted(model.teams))
            print("home:", repr(r["home_team"]))
            print("away:", repr(r["away_team"]))
            print("home in model:", r["home_team"] in model.teams)
            print("away in model:", r["away_team"] in model.teams)
            P = score_matrix(model, r["home_team"], r["away_team"], feat_vec=X)
            preds.append(market_probs(P))
        preds = pd.DataFrame(preds).reset_index(drop=True)
        val = val.reset_index(drop=True)

        y_home = val["home_win"].values
        p_home = preds["p_home"].values

        ll = log_loss(y_home, p_home)
        base = y_home.mean()
        ll_base = log_loss(y_home, np.full_like(p_home, base))
        acc = accuracy_score(y_home, (p_home > 0.5).astype(int))

        y_draw = (val["home_goals_60"] == val["away_goals_60"]).astype(int)
        p_draw = preds["p_draw_60"].values
        ll_draw = log_loss(y_draw, np.clip(p_draw, 1e-6, 1-1e-6))

        y_over55 = (val["home_goals_60"] + val["away_goals_60"] > 5.5).astype(int)
        p_over55 = preds["p_over_5.5"].values
        ll_over = log_loss(y_over55, np.clip(p_over55, 1e-6, 1-1e-6))

        print(f"\n=== {season} ({len(val)} matchs) ===")
        print(f"Home win  : LL={ll:.4f}  base={ll_base:.4f}  acc={acc:.3f}")
        print(f"Draw 60   : LL={ll_draw:.4f}")
        print(f"Over 5.5  : LL={ll_over:.4f}")


if __name__ == "__main__":
    main()