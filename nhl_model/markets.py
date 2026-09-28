# nhl_model/markets.py
import numpy as np
from scipy.stats import poisson
from .model import dc_tau, PoissonModel


def _team_index(model, team):
    return model.teams.index(team)


def lambdas_with_features(model: PoissonModel,
                          home_team, away_team,
                          feat_vec: np.ndarray | None = None):
    h = _team_index(model, home_team)
    a = _team_index(model, away_team)

    eta_h = 0.0
    eta_a = 0.0
    if feat_vec is not None:
        xs = (feat_vec - model.feat_mean) / model.feat_std
        eta_h = float(xs @ model.beta_home)
        eta_a = float(xs @ model.beta_away)

    lh = np.exp(model.mu + model.att[h] - model.dfn[a] + model.gamma + eta_h)
    la = np.exp(model.mu + model.att[a] - model.dfn[h] + eta_a)
    return float(np.clip(lh, 0.05, 12)), float(np.clip(la, 0.05, 12))


def score_matrix(model, home_team, away_team,
                 feat_vec=None, max_goals=10):
    lh, la = lambdas_with_features(model, home_team, away_team, feat_vec)
    P = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            P[i, j] = (poisson.pmf(i, lh) * poisson.pmf(j, la)
                       * dc_tau(i, j, lh, la, model.rho))
    return P / P.sum()


def market_probs(P, lines=(4.5, 5.5, 6.5)):
    n = P.shape[0]
    out = {}
    out["p_home"] = float(sum(P[i,j] for i in range(n) for j in range(n) if i > j))
    out["p_away"] = float(sum(P[i,j] for i in range(n) for j in range(n) if i < j))
    out["p_draw_60"] = float(sum(P[i,i] for i in range(n)))

    out["p_home_-1.5"] = float(sum(P[i,j] for i in range(n) for j in range(n) if i - j >= 2))
    out["p_away_+1.5"] = 1 - out["p_home_-1.5"]
    out["p_away_-1.5"] = float(sum(P[i,j] for i in range(n) for j in range(n) if j - i >= 2))
    out["p_home_+1.5"] = 1 - out["p_away_-1.5"]

    for L in lines:
        out[f"p_over_{L}"] = float(sum(P[i,j] for i in range(n) for j in range(n) if i + j > L))
        out[f"p_under_{L}"] = 1 - out[f"p_over_{L}"]
    return out