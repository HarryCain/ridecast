# Understand and demonstrate Ridecast

## The 20-second explanation

“Ridecast predicts the number of bike rentals in the next hour. I built a pipeline
that uses historical demand and weather, evaluates only on later dates, compares
against simple baselines, and shows errors and uncertainty in a dashboard.”

## What the tools do

| Tool | Plain-language role |
| --- | --- |
| Python | Runs the data preparation, model and app |
| pandas / NumPy | Organize hourly records and calculate historical features |
| scikit-learn | Fits boosted decision trees and measures prediction errors |
| Streamlit | Turns Python outputs into an interactive browser dashboard |
| Plotly | Draws the charts and hover details |
| pytest / GitHub Actions | Check that the logic keeps working locally and on GitHub |

## Follow one prediction

Imagine predicting rentals for Tuesday at 9 AM. At that point, Tuesday 8 AM's
count and weather are known. Yesterday at 9 AM and last Tuesday at 9 AM are also
known. Tuesday 9 AM's actual rentals and observed weather are still unknown.

`features.py` builds only those available inputs. `model.py` uses them to produce
a count prediction. `pipeline.py` compares the prediction to the later observed
count. `app.py` shows the result and lets you inspect the mistake.

## The most important engineering decision

Missing hours are a trap. If an hour is absent, shifting down 24 rows reaches
more than 24 clock hours into the past. Ridecast first creates an actual hourly
grid, leaving unknown hours empty, and then shifts. Tests deliberately remove
an hour and check this behavior.

Another trap is target leakage: `casual + registered = cnt`. Including either
current-hour rental component would reveal part of the answer. The input columns
are explicitly listed, and a test changes current and future outcomes and weather
to prove the current input features do not change.

## A three-minute demo

1. Open Forecast explorer. Explain MAE as “the average size of the model's hourly mistake, in rentals.” Use the generated results rather than saying “accuracy.”
2. Change the week and enable the weekly baseline. Find one peak the model misses. Explain that this is a sequence of separate one-hour forecasts.
3. Open Model audit. Point to the nonoverlapping chronological splits, both baselines, and earlier expanding-window folds.
4. Compare the no-weather ablation. Explain that more features do not automatically improve a model.
5. Show nominal versus observed interval coverage and state why those can differ.
6. If trained locally, open Forecast lab. Change a prior-hour input and run real inference. Explain why that change does not prove a causal relationship.

## Questions to be ready for

**Why not randomly split the data?** A random split mixes earlier and later periods.
For forecasting, I want to measure performance on a period that occurs after the
training data, matching the direction of actual use.

**Why gradient boosting instead of a neural network?** This is a modest tabular
dataset with calendar and historical features. Boosted trees provide a useful,
fast baseline model without needing a large neural architecture. I have not
established that it is the globally best model.

**Why Poisson loss?** Rentals are nonnegative counts. This loss uses positive
predictions and is a reasonable starting point for count outcomes. It does not
mean demand follows an exact Poisson process or that overdispersion disappears.

**What's your baseline?** Last hour's demand and the same hour last week. Both are
simple, plausible competitors. I report both on the same timestamps and describe
improvement against the stronger held-out baseline.

**Do you use test answers as inputs?** Once an earlier test hour has passed, its
outcome can be known before the next forecast. That matches a rolling one-hour
service. It would be invalid for predicting the whole test period in one batch
at its start. The tree weights are never retrained on the test period.

**Are 90% intervals guaranteed?** No. The interval radius is calibrated on an earlier
period using an absolute-residual order statistic. Seasonal changes and temporal
dependence can break coverage assumptions, so I report actual coverage and width.

**What is permutation importance?** Shuffle a feature, predict again, and see how
much the error increases. If error rises, the model relied on that information.
Correlated inputs and unrealistic shuffled histories make it an imperfect diagnostic.

**What would you improve next?** Compare variable-width quantile intervals, evaluate
multiple later seasons with a retraining schedule, add fresh data ingestion and
observability, and measure reporting latency before claiming operational usefulness.

## Make the project yours

Read `features.py` and its leakage tests first, run the project yourself, and be
able to explain one prediction end to end. Then make an improvement you can defend:
add a daily baseline, compare a simpler regression, or measure errors by weather.
Use measured numbers in resume bullets. Do not describe this offline experiment
as a deployed production service or claim it has real users.
