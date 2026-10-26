# Ridecast model card

## Intended task

Predict system-wide rental count in hour `t`, immediately before that hour, using
calendar information and observations through `t-1`. Intended for an educational
portfolio and reproducible historical evaluation. Not a station rebalancing system.

## Data

UCI Bike Sharing hourly CSV, Capital Bikeshare, 2011–2012. The actual download has
17,379 hourly records; the authoritative count for each run is computed from the CSV.
Local clock gaps are preserved as unknown values. Naive timestamps do not allow
fully resolving daylight-saving-time ambiguities. Missing source hours and eligible
sample exclusions are counted in the generated report.

Data quality checks validate the schema, nonnegative reconciled counts, normalized
weather ranges, integer hours, complete input rows, and unique local timestamps.
SHA-256 records detect differences between datasets across runs, but are provenance
fingerprints rather than a pre-pinned download-integrity guarantee.

## Inputs and availability

17 features: cyclical hour, day of week, month, known holiday/working-day flags,
elapsed days from a fixed origin, demand lags at 1/2/24/168 hours, mean and standard
deviation of the preceding 24 hours, and previous-hour temperature, humidity,
wind speed and weather category. Rolling features need at least 18 of 24 known hours;
the individual lag and weather features must all be available.

Current-hour demand components, current-hour weather, source row ID, and the
target itself are excluded. Calendar attributes are assumed known in advance.
Weather values and counts from the previous hour are assumed available without
reporting latency; real data collection may violate that assumption.

## Training and evaluation

Histogram gradient boosting with Poisson loss, 240 iterations, learning rate 0.06,
23 leaf nodes, L2 regularization 1.0, random state 42. Early stopping is disabled.
These settings are fixed; there is no test-driven hyperparameter selection.
Three expanding-window backtests run only within the training period.

70/15/15 chronological split by eligible rows. The calibration period is used
for interval calibration and feature diagnostics, not fitting tree weights.
Final test metrics are MAE, RMSE, and R². R² is not an accuracy percentage.
Previous-hour and previous-week baselines use the same evaluated timestamps.
The weather ablation retains the same rows, parameters, and test window.

This is a rolling observational backtest with frozen weights. Observed outcomes
earlier in a holdout period may feed features later in that period. This is legitimate
for one-hour-ahead forecasting and would be leakage for an entire holdout forecast
issued at a single origin. The backtest gap is 24 eligible observations, not
necessarily 24 elapsed hours because missing/ineligible observations are removed.

## Uncertainty

The absolute calibration residuals are sorted; the global radius is the
`ceil((n + 1) × 0.90)`-th order statistic, clamped to `n` for small samples.
Intervals are prediction ± radius with a lower bound of zero. This follows a
split-conformal calibration recipe, but time dependence and distribution shift
invalidate a general exchangeability-based coverage guarantee. Actual test
coverage and mean width are always reported. A constant-width band is unlikely
to capture changing peak-hour uncertainty well.

## Diagnostics and limits

Permutation importance is measured on calibration data, with three repeats and
increased MAE as the score. Correlated features and disrupted temporal relationships
limit interpretation. Sensitivity controls can produce inconsistent lag histories;
they describe model responses and establish no causal effect or scenario accuracy.

Historical results do not establish present-day performance. Seasonal shifts,
unobserved events, system growth, unavailable station-level information, reporting
delays, missing-hour selection bias, and dataset age limit practical use. A real
service would need live ingestion, latency checks, drift monitoring, retraining,
security, operator validation, and a broader evaluation across seasons/cities.

Results, dates, exact samples, and source fingerprint: [RESULTS.md](RESULTS.md)
and `artifacts/summary.json`.
