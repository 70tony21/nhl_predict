import json, pandas as pd
from nhl_model.model import DixonColesParams
from nhl_model.markets import market_probs, score_matrix
from nhl_model.config import OUT_DIR

def load_params():
    with open(OUT_DIR / "dc_params.json") as f:
        d = json.load(f)
    teams = d["teams"]
    return DixonColesParams(
        teams=teams,
        mu=d["mu"], gamma=d["gamma"], rho=d["rho"],
        att=np.array([d["att"][t] for t in teams]),
        dfn=np.array([d["dfn"][t] for t in teams]),
    )

if __name__ == "__main__":
    import numpy as np
    p = load_params()
    for home, away in [("BOS","MTL"), ("EDM","CGY")]:
        P = score_matrix(p, home, away)
        print(home, "vs", away, market_probs(P))