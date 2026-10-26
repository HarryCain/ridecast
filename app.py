"""A local, interactive model evaluation workbench. Run: streamlit run app.py."""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
GREEN, INK, GRAY = "#16705C", "#253746", "#94A3B8"
st.set_page_config(page_title="Ridecast · Demand intelligence", page_icon="🚲", layout="wide")
st.markdown("""<style>
.block-container {max-width:1420px;padding-top:2.4rem;padding-bottom:3rem}
h1 {letter-spacing:-.05em;font-weight:750!important}
h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:white;border:1px solid #E2E7EB;border-radius:12px;padding:18px 20px}
[data-testid="stMetricLabel"] {color:#637581}
[data-testid="stMetricValue"] {font-size:2rem;letter-spacing:-.04em}
[data-testid="stSidebar"] {border-right:1px solid #E2E7EB}
.eyebrow {color:#16705C;font-size:11px;letter-spacing:2px;font-weight:750;margin-bottom:10px}
.lede {color:#637581;font-size:16px;max-width:760px;line-height:1.6;margin-bottom:28px}
.badge {display:inline-block;background:#E5F2EC;border:1px solid #CCE2D7;border-radius:20px;padding:5px 12px;color:#16705C;font-size:12px}
</style>""", unsafe_allow_html=True)


@st.cache_data
def read_artifacts(modified):
    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    predictions = pd.read_csv(ARTIFACTS / "predictions.csv", index_col="timestamp", parse_dates=True)
    importance = pd.read_csv(ARTIFACTS / "importance.csv")
    return summary, predictions, importance


def style_chart(chart, height=380):
    chart.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=15, t=25, b=15),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Arial, sans-serif", color=INK, size=12),
                        legend=dict(orientation="h", y=1.15, x=0), hovermode="x unified")
    chart.update_xaxes(showgrid=False, zeroline=False)
    chart.update_yaxes(gridcolor="#E7EBEF", zeroline=False)
    return chart


st.sidebar.markdown("## ◉ Ridecast")
st.sidebar.caption("DEMAND INTELLIGENCE")
st.sidebar.divider()
page = st.sidebar.radio("Workspace", ["Forecast explorer", "Model audit", "Forecast lab"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**One hour ahead.**\n\nA transparent forecasting experiment for urban bike demand.")
st.sidebar.caption("CAPITAL BIKESHARE · 2011–2012\n\nHistorical research data. Not a live service.")
st.sidebar.link_button("Source code ↗", "https://github.com/HarryCain/ridecast", width="stretch")

if not (ARTIFACTS / "summary.json").exists():
    st.title("Your next ride starts with a forecast.")
    st.info("Train the model once to generate the dashboard's real evaluation data.")
    st.code("python -m ridecast train")
    st.stop()

summary, predictions, importance = read_artifacts((ARTIFACTS / "summary.json").stat().st_mtime_ns)
score = summary["metrics"]["Gradient boosting"]
baseline_name = min(["Previous hour", "Previous week"], key=lambda name: summary["metrics"][name]["mae"])
improvement = (1 - score["mae"] / summary["metrics"][baseline_name]["mae"]) * 100

st.markdown('<div class="eyebrow">MOBILITY / MACHINE LEARNING</div>', unsafe_allow_html=True)
if page == "Forecast explorer":
    title, badge = st.columns([4, 1])
    with title:
        st.title("A clearer view of the next hour.")
    with badge:
        st.markdown('<div class="badge">● Historical holdout evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="lede">Explore real demand, model forecasts, and the uncertainty between them. Every prediction uses information available before its target hour.</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Mean absolute error", f"{score['mae']:.1f}")
    a.caption("rentals / hour")
    b.metric("MAE reduction", f"{improvement:.1f}%")
    b.caption(f"vs. {baseline_name.lower()}")
    c.metric("Interval coverage", f"{summary['test_coverage']:.1%}")
    c.caption("90% nominal target")
    d.metric("Holdout hours", f"{len(predictions):,}")
    d.caption("chronological evaluation")
    st.write("")
    st.subheader("Demand, hour by hour")
    left, right = st.columns([3, 1])
    weeks = sorted(set(predictions.index.to_period("W-SUN").start_time))
    week = left.selectbox("Week starting", weeks, format_func=lambda value: value.strftime("%b %d, %Y"), index=min(1, len(weeks)-1))
    baseline = right.toggle("Show weekly baseline", value=False)
    view = predictions.loc[(predictions.index >= week) & (predictions.index < week + pd.Timedelta(days=7))]
    # Explicit gaps keep the chart from drawing a continuous line over excluded hours.
    view = view.reindex(pd.date_range(view.index.min(), view.index.max(), freq="h"))
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=view.index, y=view.upper, line=dict(width=0), showlegend=False, hoverinfo="skip", connectgaps=False))
    chart.add_trace(go.Scatter(x=view.index, y=view.lower, line=dict(width=0), fill="tonexty", fillcolor="rgba(22,112,92,.10)", name="90% nominal interval", hoverinfo="skip"))
    chart.add_trace(go.Scatter(x=view.index, y=view.actual, name="Observed demand", line=dict(color=INK, width=2)))
    chart.add_trace(go.Scatter(x=view.index, y=view.prediction, name="ML forecast", line=dict(color=GREEN, width=2)))
    if baseline:
        chart.add_trace(go.Scatter(x=view.index, y=view.previous_week, name="Previous week", line=dict(color=GRAY, width=1.3, dash="dot")))
    chart.update_yaxes(title="Rentals / hour")
    st.plotly_chart(style_chart(chart), width="stretch")
    st.caption("Each point is a separate one-hour-ahead forecast. The frozen model observes earlier outcomes as time advances; this is not a seven-day forecast issued at the start of the week. Missing/ineligible hours are excluded.")
    left, right = st.columns([1, 1], gap="large")
    residual = predictions.assign(error=(predictions.actual - predictions.prediction).abs())
    with left:
        st.subheader("When does the model struggle?")
        hourly = residual.groupby(residual.index.hour).error.mean()
        chart = go.Figure(go.Bar(x=hourly.index, y=hourly.values, marker_color=GREEN))
        chart.update_xaxes(title="Hour of day", dtick=3)
        chart.update_yaxes(title="Mean absolute error")
        st.plotly_chart(style_chart(chart, 270), width="stretch")
    with right:
        st.subheader("Largest misses this week")
        misses = view.assign(error=(view.actual - view.prediction).abs()).nlargest(5, "error")
        display = misses[["actual", "prediction", "error"]].round(1).rename(columns={"actual":"Observed", "prediction":"Forecast", "error":"Abs. error"})
        st.dataframe(display, width="stretch")
        st.caption("Large misses help identify unusual events and places where a global uncertainty band is too narrow.")
    st.download_button("Download holdout predictions ↓", (ARTIFACTS / "predictions.csv").read_bytes(), "ridecast-predictions.csv", "text/csv")

elif page == "Model audit":
    st.title("Good forecasts need honest evaluation.")
    st.markdown('<div class="lede">Trace the experiment from source data to chronological evaluation. Baselines, missing hours, and model limitations stay visible.</div>', unsafe_allow_html=True)
    st.subheader("01 / The evaluation contract")
    cols = st.columns(3)
    for col, (name, value) in zip(cols, summary["splits"].items()):
        with col:
            st.metric(name.capitalize(), f"{value['rows']:,} hours")
            st.caption(f"{value['start'][:10]} → {value['end'][:10]}")
    st.info("Train first 70% → calibrate on next 15% → evaluate on final 15%. No shuffled split, current-hour weather, or casual/registered rental counts enter the model. Early stopping is disabled to avoid an internal random validation split.")
    st.subheader("02 / Compare before claiming improvement")
    table = pd.DataFrame(summary["metrics"]).T.rename(columns={"mae":"MAE ↓", "rmse":"RMSE ↓", "r2":"R² ↑"})
    st.dataframe(table.round(3), width="stretch")
    st.caption("Without weather: retrain with the same parameters and samples, removing only previous-hour weather. Differences describe this experiment; they do not prove causation.")
    st.subheader("03 / Performance across earlier time windows")
    if summary["backtests"]:
        folds = pd.DataFrame(summary["backtests"])
        st.dataframe(folds[["fold", "train_end", "test_start", "test_end", "rows", "mae", "previous_hour_mae", "previous_week_mae"]].round(2), hide_index=True, width="stretch")
        st.caption("Three expanding-window backtests inside training data, with a 24-sample gap. Each holdout still uses rolling one-hour-ahead observations. Parameters are fixed; these folds were not used for a search.")
    st.subheader("04 / What information does the model use?")
    selected = importance.head(9).iloc[::-1]
    chart = go.Figure(go.Bar(x=selected.mae_increase, y=selected.feature, orientation="h", marker_color=GREEN,
                            error_x=dict(type="data", array=selected["std"])))
    chart.update_xaxes(title="Increase in calibration MAE after shuffling")
    st.plotly_chart(style_chart(chart, 340), width="stretch")
    st.caption("Permutation importance on calibration data; three repeats. Correlated lag features can share importance. Shuffling breaks time structure, so use this as a diagnostic rather than a causal explanation.")
    with st.expander("Data provenance and limitations", expanded=True):
        st.write(f"{summary['raw_rows']:,} source rows · {summary['missing_hours']} missing clock hours · {summary['excluded_observed_rows']:,} observed rows excluded for unavailable history.")
        st.write("Missing hours remain unknown, rather than becoming zero rentals. Lags are calculated on an hourly clock grid. The dataset provides naive local timestamps; daylight-saving ambiguities cannot be fully resolved.")
        st.write(f"Nominal 90% intervals achieved {summary['test_coverage']:.1%} coverage, with a mean width of {summary['average_interval_width']:.1f} rentals. Seasonal shift and dependent observations mean coverage is not guaranteed.")
        st.write("This forecasts system-wide hourly rentals, not demand at a specific station. The data is historical; deployment would need fresh data, availability checks, monitoring, and retraining.")
        st.markdown("Data: [Fanaee-T (2013), UCI Bike Sharing](https://doi.org/10.24432/C5W894) · CC BY 4.0")
        st.code(f"SHA-256: {summary['data_sha256']}", language=None)

else:
    st.title("Put a forecast under the microscope.")
    st.markdown('<div class="lede">Start from a real historical hour, change its prior-hour inputs, and run the trained model. Explore model sensitivity without treating it as a causal simulator.</div>', unsafe_allow_html=True)
    if not (ARTIFACTS / "model.joblib").exists() or not (ARTIFACTS / "features.csv").exists():
        st.info("Run `python -m ridecast train` locally to enable inference. The explorer and audit work from the committed results.")
        st.stop()
    # Only load artifacts generated by this project's local training command.
    @st.cache_resource
    def load_model(modified):
        return joblib.load(ARTIFACTS / "model.joblib")
    bundle = load_model((ARTIFACTS / "model.joblib").stat().st_mtime_ns)
    features = pd.read_csv(ARTIFACTS / "features.csv", index_col="timestamp", parse_dates=True)
    stamp = st.selectbox("Historical target hour", list(features.index), format_func=lambda value: value.strftime("%a, %b %d · %H:%M"))
    row = features.loc[[stamp]].copy()
    with st.form("forecast"):
        left, right = st.columns(2)
        last_hour = left.number_input("Rentals in previous hour", min_value=0, max_value=2000, value=int(row.iloc[0].lag_1h))
        temperature = right.number_input("Previous-hour temperature (°C)", min_value=-8.0, max_value=39.0,
                                         value=round(float(row.iloc[0].temperature_previous * 47 - 8), 2), step=.5)
        st.caption("Other calendar and historical features remain at their observed values. Changing one lag can create an inconsistent history; this lab illustrates sensitivity within the model.")
        submitted = st.form_submit_button("Run forecast →", type="primary")
    if submitted:
        row["lag_1h"] = last_hour
        row["temperature_previous"] = (temperature + 8) / 47
        predicted = float(bundle["model"].predict(row[bundle["features"]])[0])
        a, b, c = st.columns(3)
        a.metric("Scenario forecast", f"{predicted:.0f} rentals")
        b.metric("Original forecast", f"{predictions.loc[stamp, 'prediction']:.0f} rentals")
        c.metric("Historical observation", f"{predictions.loc[stamp, 'actual']:.0f} rentals")
        st.caption("Scenario forecasts have no measured accuracy or coverage guarantee. The calibrated interval is evaluated only on the original historical test inputs.")

st.divider()
st.caption("RIDECAST / Python · pandas · scikit-learn · Streamlit · Plotly    |    Reproducible by design. Evaluated in time order.")
