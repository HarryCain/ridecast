"""Model, metrics and calibration kept independent of the presentation layer."""
import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def estimator():
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=.06, max_iter=240, max_leaf_nodes=23,
        l2_regularization=1.0, early_stopping=False, random_state=42,
    )


def metrics(actual, predicted) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def calibration_radius(actual, predicted, coverage=.90) -> float:
    residuals = np.abs(np.asarray(actual) - np.asarray(predicted))
    if not 0 < coverage < 1 or len(residuals) == 0 or not np.isfinite(residuals).all():
        raise ValueError("Need finite calibration residuals and 0 < coverage < 1")
    # Finite-sample split-conformal order statistic, clamped for very small sets.
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * coverage))
    return float(np.sort(residuals)[rank - 1])


def interval(predicted, radius):
    point = np.maximum(0, np.asarray(predicted))
    return np.maximum(0, point - radius), point + radius
