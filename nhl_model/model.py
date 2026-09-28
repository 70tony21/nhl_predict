# nhl_model/model.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from dataclasses import dataclass, field


def dc_tau(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - lh * la * rho
    if x == 0 and y == 1: return 1 + lh * rho
    if x == 1 and y == 0: return 1 + la * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


# features utilisées dans le modèle
FEATURES = [
    "home_CF_pct_r10", "away_CF_pct_r10",
    "home_SF_pct_r10", "away_SF_pct_r10",
    "home_PDO_r10", "away_PDO_r10",
    "home_CF_per60_r10", "home_CA_per60_r10",
    "away_CF_per60_r10", "away_CA_per60_r10",
]


@dataclass
class PoissonModel:
    teams: list
    mu: float
    gamma: float
    rho: float
    att: np.ndarray
    dfn: np.ndarray
    beta_home: np.ndarray
    beta_away: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray
    features: list = field(default_factory=lambda: FEATURES)

    def to_dict(self):
        return {
            "mu": self.mu, "gamma": self.gamma, "rho": self.rho,
            "teams": self.teams,
            "att": dict(zip(self.teams, self.att.tolist())),
            "dfn": dict(zip(self.teams, self.dfn.tolist())),
            "beta_home": self.beta_home.tolist(),
            "beta_away": self.beta_away.tolist(),
            "feat_mean": self.feat_mean.tolist(),
            "feat_std": self.feat_std.tolist(),
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, d):
        teams = d["teams"]
        return cls(
            teams=teams,
            mu=d["mu"], gamma=d["gamma"], rho=d["rho"],
            att=np.array([d["att"][t] for t in teams]),
            dfn=np.array([d["dfn"][t] for t in teams]),
            beta_home=np.array(d["beta_home"]),
            beta_away=np.array(d["beta_away"]),
            feat_mean=np.array(d["feat_mean"]),
            feat_std=np.array(d["feat_std"]),
            features=d.get("features", FEATURES),
        )


def _unpack(p, n, k):
    mu, gamma = p[0], p[1]
    att = p[2:2+n]
    dfn = p[2+n:2+2*n]
    beta_h = p[2+2*n:2+2*n+k]
    beta_a = p[2+2*n+k:2+2*n+2*k]
    rho = p[-1]
    return mu, gamma, att, dfn, beta_h, beta_a, rho


def fit(matches, max_iter=1000, reg=0.05, reg_beta=0.02):
    df = matches.dropna(subset=["home_goals_60", "away_goals_60"] + FEATURES).copy()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    t2i = {t: i for i, t in enumerate(teams)}

    hi = df["home_team"].map(t2i).values
    ai = df["away_team"].map(t2i).values
    hg = df["home_goals_60"].astype(int).values
    ag = df["away_goals_60"].astype(int).values

    X = df[FEATURES].values.astype(float)
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    Xs = (X - feat_mean) / feat_std

    n, k = len(teams), len(FEATURES)

    x0 = np.concatenate([
        [np.log(2.7), 0.1],
        np.zeros(n), np.zeros(n),
        np.zeros(k), np.zeros(k),
        [-0.03],
    ])

    bounds = (
        [(None, None), (-0.5, 0.5)]
        + [(-1.5, 1.5)] * n
        + [(-1.5, 1.5)] * n
        + [(-0.3, 0.3)] * (2 * k)
        + [(-0.08, 0.03)]
    )

    cons = [
        {"type": "eq", "fun": lambda p, n=n: np.sum(p[2:2+n])},
        {"type": "eq", "fun": lambda p, n=n: np.sum(p[2+n:2+2*n])},
    ]

    def nll(p):
        mu, gamma, att, dfn, beta_h, beta_a, rho = _unpack(p, n, k)

        eta_h = Xs @ beta_h
        eta_a = Xs @ beta_a

        log_lh = mu + att[hi] - dfn[ai] + gamma + eta_h
        log_la = mu + att[ai] - dfn[hi] + eta_a

        log_lh = np.clip(log_lh, -3.0, 2.5)
        log_la = np.clip(log_la, -3.0, 2.5)
        lh = np.exp(log_lh)
        la = np.exp(log_la)

        ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
        taus = np.array([dc_tau(int(x), int(y), l, a, rho)
                         for x, y, l, a in zip(hg, ag, lh, la)])
        taus = np.clip(taus, 1e-6, None)

        reg_term = reg * (np.sum(att**2) + np.sum(dfn**2)) \
                   + reg_beta * (np.sum(beta_h**2) + np.sum(beta_a**2))

        return -np.sum(ll + np.log(taus)) + reg_term

    res = minimize(nll, x0, method="SLSQP", bounds=bounds,
                   constraints=cons,
                   options={"maxiter": max_iter, "ftol": 1e-8})

    if not res.success:
        print("⚠ SLSQP:", res.message)

    mu, gamma, att, dfn, beta_h, beta_a, rho = _unpack(res.x, n, k)
    return PoissonModel(teams, mu, gamma, rho, att, dfn,
                        beta_h, beta_a, feat_mean, feat_std)