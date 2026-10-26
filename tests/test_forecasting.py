import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ridecast.data import load, validate
from ridecast.features import FEATURES, chronological_split, make_features
from ridecast.model import calibration_radius, interval
from ridecast.pipeline import train


@pytest.fixture
def raw():
    index = pd.date_range("2011-01-01", periods=900, freq="h")
    count = np.round(90 + 50 * np.sin(index.hour * np.pi / 12) + np.arange(900) / 30).astype(int)
    return pd.DataFrame({"cnt":count, "casual": count//3, "registered": count-count//3,
        "temp":.5, "hum":.6, "windspeed":.2, "weathersit":1,
        "holiday":0, "workingday":(index.day_of_week < 5).astype(int),
        "dteday":index.strftime("%Y-%m-%d"), "hr":index.hour}, index=index)


def test_current_and_future_outcomes_do_not_change_current_features(raw):
    timestamp = raw.index[450]
    expected = make_features(raw).loc[:timestamp, FEATURES]
    changed = raw.copy()
    changed.loc[timestamp:, ["cnt", "casual", "registered", "temp", "hum", "windspeed", "weathersit"]] = 999
    pd.testing.assert_frame_equal(make_features(changed).loc[:timestamp, FEATURES], expected)


def test_lags_use_elapsed_hours_not_observation_rows(raw):
    missing = raw.index[300]
    actual = make_features(raw.drop(missing))
    assert missing + pd.Timedelta(hours=24) not in actual.index
    stamp = raw.index[400]
    assert actual.loc[stamp, "lag_24h"] == raw.loc[stamp - pd.Timedelta(hours=24), "cnt"]


def test_rolling_window_excludes_current_target(raw):
    frame = make_features(raw)
    stamp = raw.index[450]
    assert frame.loc[stamp, "mean_24h"] == pytest.approx(raw.cnt.iloc[426:450].mean())
    assert "casual" not in FEATURES and "registered" not in FEATURES and "target" not in FEATURES


def test_splits_are_disjoint_ordered_and_complete(raw):
    frame = make_features(raw)
    training, calibration, test = chronological_split(frame)
    assert training.index.max() < calibration.index.min() < test.index.min()
    assert calibration.index.max() < test.index.min()
    assert len(training) + len(calibration) + len(test) == len(frame)
    with pytest.raises(ValueError):
        chronological_split(frame.iloc[::-1])


def test_conformal_uses_finite_sample_order_statistic():
    assert calibration_radius(np.arange(1, 11), np.zeros(10), .8) == 9
    with pytest.raises(ValueError):
        calibration_radius([], [])
    with pytest.raises(ValueError):
        calibration_radius([1], [np.nan])


def test_intervals_are_nonnegative_and_contain_prediction():
    lower, upper = interval([0, 2, 100], 10)
    assert lower.tolist() == [0, 0, 90]
    assert upper.tolist() == [10, 12, 110]


def test_source_count_reconciliation_is_required(raw):
    validate(raw)
    corrupt = raw.copy()
    corrupt.iloc[0, corrupt.columns.get_loc("cnt")] += 1
    with pytest.raises(ValueError, match="reconcile"):
        validate(corrupt)


def test_duplicate_timestamps_are_rejected(raw):
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([raw, raw.iloc[:1]]))


def test_pipeline_roundtrip_and_artifact_consistency(raw, tmp_path):
    source = tmp_path / "hour.csv"
    raw.to_csv(source, index=False)
    summary = train(tmp_path, source, backtest=False)
    stored = json.loads((tmp_path / "artifacts/summary.json").read_text())
    assert stored == summary
    predictions = pd.read_csv(tmp_path / "artifacts/predictions.csv")
    features = pd.read_csv(tmp_path / "artifacts/features.csv")
    bundle = joblib.load(tmp_path / "artifacts/model.joblib")
    np.testing.assert_allclose(bundle["model"].predict(features[FEATURES]), predictions.prediction)
    assert summary["metrics"]["Gradient boosting"]["mae"] == pytest.approx(np.abs(predictions.actual-predictions.prediction).mean())
    assert len(predictions) == summary["splits"]["test"]["rows"]
    assert summary["test_coverage"] == pytest.approx(((predictions.actual >= predictions.lower) & (predictions.actual <= predictions.upper)).mean())
    assert (tmp_path / "docs/RESULTS.md").exists()
