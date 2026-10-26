"""Reproduce the experiment, temporal backtests, ablation and report artifacts."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from .data import CITATION, SOURCE, download, load
from .features import FEATURES, WEATHER_FEATURES, chronological_split, make_features
from .model import calibration_radius, estimator, interval, metrics


def train(root: Path, csv: Path | None = None, backtest=True) -> dict:
    csv = csv or download(root / "data/raw")
    raw = load(csv)
    frame = make_features(raw)
    training, calibration, test = chronological_split(frame)
    model = estimator().fit(training[FEATURES], training.target)
    calibration_prediction = model.predict(calibration[FEATURES])
    radius = calibration_radius(calibration.target, calibration_prediction)
    prediction = model.predict(test[FEATURES])
    lower, upper = interval(prediction, radius)
    scores = {"Gradient boosting": metrics(test.target, prediction),
              "Previous hour": metrics(test.target, test.lag_1h),
              "Previous week": metrics(test.target, test.lag_168h)}
    no_weather = [name for name in FEATURES if name not in WEATHER_FEATURES]
    ablation = estimator().fit(training[no_weather], training.target)
    scores["Without weather"] = metrics(test.target, ablation.predict(test[no_weather]))
    predictions = pd.DataFrame({"actual": test.target, "prediction": prediction,
                                "lower": lower, "upper": upper,
                                "previous_hour": test.lag_1h, "previous_week": test.lag_168h})
    folds = []
    if backtest:
        for number, (fit, holdout) in enumerate(TimeSeriesSplit(n_splits=3, gap=24).split(training), 1):
            fit_data, check = training.iloc[fit], training.iloc[holdout]
            fold_model = estimator().fit(fit_data[FEATURES], fit_data.target)
            folds.append({"fold": number, "train_end": str(fit_data.index.max()),
                          "test_start": str(check.index.min()), "test_end": str(check.index.max()),
                          "rows": len(check), **metrics(check.target, fold_model.predict(check[FEATURES])),
                          "previous_hour_mae": metrics(check.target, check.lag_1h)["mae"],
                          "previous_week_mae": metrics(check.target, check.lag_168h)["mae"]})
    # Diagnostics on calibration only: holdout test remains for final evaluation.
    importance = permutation_importance(model, calibration[FEATURES], calibration.target,
                                      scoring="neg_mean_absolute_error", n_repeats=3, random_state=42)
    importance_frame = pd.DataFrame({"feature": FEATURES, "mae_increase": importance.importances_mean,
                                    "std": importance.importances_std}).sort_values("mae_increase", ascending=False)
    splits = {name: {"start": str(part.index.min()), "end": str(part.index.max()), "rows": len(part)}
              for name, part in [("train", training), ("calibration", calibration), ("test", test)]}
    total_hours = len(pd.date_range(raw.index.min(), raw.index.max(), freq="h"))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE, "citation": CITATION, "license": "CC BY 4.0",
        "data_sha256": sha256(csv.read_bytes()).hexdigest(), "raw_rows": len(raw),
        "missing_hours": total_hours - len(raw), "eligible_rows": len(frame),
        "excluded_observed_rows": len(raw) - len(frame), "feature_count": len(FEATURES),
        "splits": splits, "metrics": scores, "nominal_coverage": .90,
        "interval_radius": radius,
        "test_coverage": float(np.mean((test.target >= lower) & (test.target <= upper))),
        "average_interval_width": float(np.mean(upper - lower)),
        "parameters": model.get_params(), "backtests": folds,
        "evaluation": "Rolling one-hour-ahead. Prior test outcomes become observable before subsequent forecasts. Model weights remain frozen.",
    }
    output = root / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv")
    test[FEATURES].to_csv(output / "features.csv")
    importance_frame.to_csv(output / "importance.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "backtests.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    joblib.dump({"model": model, "features": FEATURES, "radius": radius}, output / "model.joblib")
    (output / "environment.json").write_text(json.dumps({"python": platform.python_version(),
        "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}, indent=2) + "\n")
    write_report(root, summary)
    return summary


def write_report(root: Path, summary: dict):
    scores = summary["metrics"]
    model = scores["Gradient boosting"]
    baseline_name = min(["Previous hour", "Previous week"], key=lambda name: scores[name]["mae"])
    improvement = (1 - model["mae"] / scores[baseline_name]["mae"]) * 100
    weather_change = scores["Without weather"]["mae"] - model["mae"]
    table = "\n".join(f"| {name} | {value['mae']:.2f} | {value['rmse']:.2f} | {value['r2']:.3f} |" for name, value in scores.items())
    text = f"""# Measured experiment results

Generated by `python -m ridecast train`; values come from this run, not illustrative data.

{summary['raw_rows']:,} observed hours; {summary['missing_hours']} missing clock hours;
{summary['eligible_rows']:,} eligible samples after past-only feature construction.
The final chronological test contains {summary['splits']['test']['rows']:,} hours,
from {summary['splits']['test']['start']} to {summary['splits']['test']['end']}.

| Model | MAE (rentals/hour) | RMSE | R² |
| --- | ---: | ---: | ---: |
{table}

The ML model changes MAE by **{improvement:.1f}% lower** versus the stronger test baseline
({baseline_name}). Both baseline scores are reported; this comparator is descriptive,
not a model-selection decision. Negative improvement would mean the model is worse.

Nominal 90% calibration intervals cover **{summary['test_coverage']:.1%}** of test observations;
mean width is **{summary['average_interval_width']:.1f} rentals**. Temporal dependence and seasonal
shift prevent treating nominal coverage as a guarantee. The global radius is
{summary['interval_radius']:.2f}; lower endpoints are clipped at zero.

## Ablation finding

Removing prior-hour weather changes test MAE by **{weather_change:+.2f} rentals/hour**
relative to the full model (negative means the simpler model did better).
This finding is reported as an ablation; the primary model was not switched after
seeing test performance. A future comparison should choose inputs on earlier
validation folds and evaluate on a fresh holdout.

## Resume wording supported by this run

**Ridecast — Bike Demand Forecasting | Python, pandas, scikit-learn, Streamlit, Plotly**

- Built a reproducible hourly demand forecasting pipeline over {summary['raw_rows']:,} real observations, engineering {summary['feature_count']} calendar, lag and historical-weather features with timestamp-aware leakage checks.
- Reduced held-out MAE by {improvement:.1f}% versus the stronger of persistence and weekly seasonal baselines, achieving {model['mae']:.1f} rentals/hour MAE on {summary['splits']['test']['rows']:,} chronological test samples.
- Delivered an interactive evaluation dashboard with expanding-window backtests, weather-feature ablation, permutation diagnostics and calibrated prediction intervals.

Use the results from your own rerun if they differ. See `docs/INTERVIEW_GUIDE.md`
for the assumptions, limitations and code walkthrough. This is an offline portfolio
experiment, with no claim of live deployment or real operator use.
"""
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/RESULTS.md").write_text(text)
