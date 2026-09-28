import numpy as np
from sklearn.metrics import log_loss, brier_score_loss

def eval_binary(y_true, p_pred, label="model"):
    return {
        "log_loss": log_loss(y_true, np.clip(p_pred, 1e-6, 1-1e-6)),
        "brier":    brier_score_loss(y_true, p_pred),
        "acc":      ((p_pred > 0.5) == y_true).mean(),
        "n":        len(y_true),
    }

def closing_line_value(p_model, odds_close, side="home"):
    """CLV simplifiée : EV du pari à la cote de fermeture."""
    return p_model * odds_close - 1