"""Predict hour t using data through t-1; never use t's rental components or weather."""
import numpy as np
import pandas as pd

FEATURES = [
    "hour_sin", "hour_cos", "weekday", "month", "workingday", "holiday", "trend_days",
    "lag_1h", "lag_2h", "lag_24h", "lag_168h", "mean_24h", "std_24h",
    "temperature_previous", "humidity_previous", "wind_previous", "weather_previous",
]
WEATHER_FEATURES = [name for name in FEATURES if name.endswith("_previous")]


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or not isinstance(raw.index, pd.DatetimeIndex) or raw.index.has_duplicates:
        raise ValueError("Expected nonempty data indexed by unique timestamps")
    # Missing records represent unknown demand, never zero demand. Reindex FIRST:
    # shift(24) must mean 24 elapsed hours, not 24 available observations.
    grid = raw.sort_index().reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="h"))
    result = pd.DataFrame(index=grid.index)
    result.index.name = "timestamp"
    result["hour_sin"] = np.sin(2 * np.pi * grid.index.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * grid.index.hour / 24)
    result["weekday"] = grid.index.day_of_week
    result["month"] = grid.index.month
    # The origin is fixed, rather than fitted using future data.
    result["trend_days"] = (grid.index - pd.Timestamp("2011-01-01")).total_seconds() / 86400
    result["workingday"] = grid.workingday
    result["holiday"] = grid.holiday
    for hours in [1, 2, 24, 168]:
        result[f"lag_{hours}h"] = grid.cnt.shift(hours)
    history = grid.cnt.shift(1)
    result["mean_24h"] = history.rolling(24, min_periods=18).mean()
    result["std_24h"] = history.rolling(24, min_periods=18).std()
    for source, target in [("temp", "temperature_previous"), ("hum", "humidity_previous"),
                           ("windspeed", "wind_previous"), ("weathersit", "weather_previous")]:
        result[target] = grid[source].shift(1)
    result["target"] = grid.cnt
    # Retain exactly the same complete rows for ML, ablation and both baselines.
    return result.dropna(subset=FEATURES + ["target"])


def chronological_split(frame: pd.DataFrame):
    if len(frame) < 100 or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Need at least 100 unique, chronologically ordered samples")
    first, second = int(len(frame) * .70), int(len(frame) * .85)
    return frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]
