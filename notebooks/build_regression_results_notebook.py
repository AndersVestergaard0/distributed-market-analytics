import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Regression Results - Realized Volatility Forecasting

INFS3208 Individual Project - Distributed Market Analytics

Loads the GBTRegressor model already trained and saved by
`src/models/regression.py` (training happens there, not here - this notebook
is the visualisation/results layer, since re-running Spark ML training on
167M raw rows from inside a notebook on every edit would be slow and hard to
version control).""")

code("""import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

sys.path.append(str(Path("../src/pipeline").resolve()))
sys.path.append(str(Path("../src/models").resolve()))

from spark_session import get_spark
from regression import FEATURE_COLS, DATA_PROCESSED, MODELS_DIR, load_split, rmspe
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator

spark = get_spark()
model = PipelineModel.load(str(MODELS_DIR / "regression_best"))
gbt = model.stages[-1]
print("Loaded model:", type(gbt).__name__, "| trees:", gbt.getNumTrees, "| maxDepth:", gbt.getOrDefault("maxDepth"))""")

md("""## 1. Metrics across all three splits

Re-running the saved model's `.transform()` (fast inference, no retraining) on
train/val/test lets us display one consistent metrics table here, rather than
copy-pasting numbers out of a terminal log.""")

code("""def evaluate_split(name):
    df = load_split(spark, name)
    pred = model.transform(df).withColumn("prediction_orig", F.exp(F.col("prediction")))
    rmse_log = RegressionEvaluator(labelCol="log_target", predictionCol="prediction", metricName="rmse").evaluate(pred)
    r2_log = RegressionEvaluator(labelCol="log_target", predictionCol="prediction", metricName="r2").evaluate(pred)
    rmspe_orig = rmspe(pred)
    return {"split": name, "rmse_log": rmse_log, "r2_log": r2_log, "rmspe_orig": rmspe_orig}, pred

rows = []
preds = {}
for split_name in ["train", "val", "test"]:
    metrics, pred_df = evaluate_split(split_name)
    rows.append(metrics)
    preds[split_name] = pred_df

metrics_table = pd.DataFrame(rows).set_index("split")
metrics_table""")

md("""## 2. Predicted vs. actual - test set

The closer points sit to the y=x line, the better. Plotted on log scale since
volatility spans multiple orders of magnitude - the same absolute distance
from the line means very different things at low vs. high volatility, so a
linear-scale plot would visually hide most of the structure.""")

code("""test_pd = preds["test"].select("target", "prediction_orig", "log_target", "prediction").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sample = test_pd.sample(n=min(15000, len(test_pd)), random_state=42)
axes[0].scatter(sample["target"], sample["prediction_orig"], s=6, alpha=0.25, color="#4C72B0")
lims = [test_pd["target"].quantile(0.001), test_pd["target"].quantile(0.999)]
axes[0].plot(lims, lims, "r--", linewidth=1, label="y = x")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Actual realized volatility (target)")
axes[0].set_ylabel("Predicted realized volatility")
axes[0].set_title("Predicted vs actual (log-log, test set)")
axes[0].legend()

resid = test_pd["log_target"] - test_pd["prediction"]
sns.histplot(resid, bins=80, ax=axes[1], color="#DD8452")
axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_title("Residuals (log-target scale, test set)")
axes[1].set_xlabel("log_target - prediction")

plt.tight_layout()
plt.show()

print("Residual mean:", resid.mean(), " | Residual std:", resid.std())""")

md("""## 3. Feature importance

Which engineered microstructure features the GBT model actually relies on to
forecast next-window volatility.""")

code("""importances = pd.Series(
    gbt.featureImportances.toArray(), index=FEATURE_COLS
).sort_values(ascending=True)

plt.figure(figsize=(7, 6))
importances.plot.barh(color="#55A868")
plt.title("GBTRegressor feature importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

importances.sort_values(ascending=False)""")

md("""## 4. Naive baseline & Diebold-Mariano significance test

A bare RMSPE number is meaningless without an anchor. This compares the
loaded GBT model against the naive persistence baseline (forecast next-window
volatility as the current window's `realized_vol_book`) and runs a
Diebold-Mariano test on the paired per-`time_id` loss series to confirm the
improvement isn't just noise - both computed by re-using the predictions
already produced by `model.transform()` above, no retraining involved.""")

code("""from regression import naive_rmspe, time_id_sq_pct_err
from stats import diebold_mariano

baseline_rows = []
for split_name in ["val", "test"]:
    pred_df = preds[split_name]
    naive = naive_rmspe(pred_df)
    model_rmspe = metrics_table.loc[split_name, "rmspe_orig"]
    baseline_rows.append({
        "split": split_name,
        "naive_rmspe": naive,
        "model_rmspe": model_rmspe,
        "pct_improvement_vs_naive": (naive - model_rmspe) / naive,
    })

baseline_table = pd.DataFrame(baseline_rows).set_index("split")
baseline_table""")

code("""model_loss = time_id_sq_pct_err(preds["test"], pred_col="prediction_orig")
naive_loss = time_id_sq_pct_err(preds["test"], pred_col="realized_vol_book")
dm_stat, dm_pvalue = diebold_mariano(model_loss, naive_loss)

print(f"Diebold-Mariano test (GBTRegressor vs. naive persistence), test set:")
print(f"  DM statistic = {dm_stat:.4f}")
print(f"  p-value      = {dm_pvalue:.6f}")
verdict = "statistically significant" if dm_pvalue < 0.05 else "not statistically significant"
print(f"  -> {verdict} improvement over naive persistence (alpha=0.05)")""")

md("""## 5. Feature ablation

Re-fits the same GBT architecture with a feature group removed, to test
whether OFI and spread/micro-price actually matter for forecasting
volatility rather than just assuming it from the narrative. Retraining two
extra models is done by `python src/models/ablation.py`, not inside this
notebook (same reason the main model isn't retrained here) - these numbers
are that run's output.""")

code("""ablation_results = pd.DataFrame([
    {"ablation": "full feature set", "test_rmspe": 0.2811, "delta_vs_full": 0.0000},
    {"ablation": "without OFI (total_ofi, avg_ofi)", "test_rmspe": 0.2802, "delta_vs_full": -0.0009},
    {"ablation": "without spread/micro-price", "test_rmspe": 0.3037, "delta_vs_full": 0.0226},
]).set_index("ablation")

plt.figure(figsize=(7, 3.5))
ablation_results["delta_vs_full"].plot.barh(color=["#55A868", "#C44E52"])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Test RMSPE delta vs. full feature set (higher = worse)")
plt.xlabel("RMSPE delta")
plt.tight_layout()
plt.show()

ablation_results""")

md("""## 6. Walk-forward fold stability

Checks whether the test RMSPE above is a lucky static split rather than a
stable result. `python src/models/split.py` bins the 3,830 distinct
`time_id`s into 5 consecutive chronological folds; `python
src/models/walk_forward.py` trains fold *i* on every `time_id` before it and
evaluates on fold *i* only (proper walk-forward, never trains on the future)
- fold 0 has no prior data so it's warm-up-only, giving 4 independent
readings. Numbers below are that run's output.""")

code("""fold_rmspe = pd.Series([0.2865, 0.2785, 0.3031, 0.2968], index=[1, 2, 3, 4], name="rmspe")
mean_rmspe, std_rmspe = fold_rmspe.mean(), fold_rmspe.std()

plt.figure(figsize=(7, 4))
fold_rmspe.plot.bar(color="#4C72B0")
plt.axhline(mean_rmspe, color="red", linestyle="--", linewidth=1, label=f"mean = {mean_rmspe:.4f}")
plt.axhspan(mean_rmspe - std_rmspe, mean_rmspe + std_rmspe, color="red", alpha=0.1, label=f"±1 std ({std_rmspe:.4f})")
plt.xlabel("walk-forward fold")
plt.ylabel("RMSPE")
plt.title("Walk-forward RMSPE by fold")
plt.legend()
plt.tight_layout()
plt.show()

print(f"RMSPE across folds: mean={mean_rmspe:.4f}  std={std_rmspe:.4f}")""")

md("""## Takeaways

- **`realized_vol_book` dominates** (~51% importance) - confirms the volatility
  clustering pattern found in the EDA (r=0.80 between current- and
  next-window volatility): past turbulence is the single strongest predictor
  of future turbulence.
- **`avg_spread_bps` is second** (~24%) - a wider spread reflects market-maker
  uncertainty, itself informative about coming volatility. The ablation
  confirms this is real signal, not noise: dropping it raises test RMSPE by
  +0.0226 (from 0.2811 to 0.3037).
- **OFI (`total_ofi`/`avg_ofi`) contributes almost nothing here** (<0.2%
  combined feature importance), and the ablation confirms it: removing OFI
  entirely *lowers* test RMSPE slightly (-0.0009) - the model isn't relying
  on it at all. Expected, since OFI is a *directional* signal (predicts price
  movement direction), not a *volatility* signal - it should matter far more
  in the classification (trade direction) task.
- The GBT model beats the naive persistence baseline by **+23.2%** RMSPE on
  the test set, and a Diebold-Mariano test confirms that improvement is
  statistically significant (DM=-14.72, p≈0.0000), not a lucky point estimate.
- Walk-forward evaluation across 4 out-of-sample folds gives RMSPE =
  0.2865 / 0.2785 / 0.3031 / 0.2968 (mean 0.2912, std 0.0109) - tightly
  clustered around the static test RMSPE of 0.2811, so the headline number
  isn't a lucky split; error is stable across the dataset.
- Predicted-vs-actual points track the y=x line reasonably closely on the
  log-log plot, and residuals are centered near zero with no strong skew
  remaining - consistent with the test R²=0.82 / RMSPE≈28% reported by the
  training script.""")

code("""spark.stop()""")

nb["cells"] = cells

with open("Regression_Results.ipynb", "w") as f:
    nbf.write(nb, f)

print("notebook written")
