"""Task 1/4 - Regression: forecast realized volatility.

Predicts log(target) (realized volatility of the *next* 10-minute window)
from the engineered microstructure features of the *current* window. Trains
two models - a linear baseline and a gradient-boosted tree model - evaluates
both on the held-out validation set, then reports final metrics on the
untouched test set for the better of the two.

Metrics:
- RMSE on the log-target (standard regression error).
- RMSPE on the original volatility scale (Optiver's own competition metric:
  root mean squared *percentage* error - appropriate here since volatility
  spans multiple orders of magnitude and we care about relative, not
  absolute, error).
"""

from pathlib import Path
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402
from paths import DATA_PROCESSED, MODELS_DIR, ensure_dir  # noqa: E402
from mongo_export import write_results  # noqa: E402
from stats import diebold_mariano  # noqa: E402

FEATURE_COLS = [
    "avg_spread_bps",
    "avg_micro_price",
    "mid_price_stddev",
    "total_ofi",
    "avg_ofi",
    "realized_vol_book",
    "book_update_count",
    "trade_volume",
    "trade_count",
    "avg_trade_size",
    "avg_order_count",
    "trade_price_stddev",
]


def load_split(spark, name):
    df = spark.read.parquet(str(DATA_PROCESSED / f"{name}.parquet"))
    return df.withColumn("log_target", F.log(F.col("target")))


def build_pipeline(regressor):
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    return Pipeline(stages=[assembler, scaler, regressor])


def rmspe(df, pred_col="prediction_orig", label_col="target"):
    """Root mean squared percentage error, Optiver's own competition metric."""
    err = df.select(
        F.pow((F.col(label_col) - F.col(pred_col)) / F.col(label_col), 2).alias("sq_pct_err")
    )
    return err.agg(F.sqrt(F.avg("sq_pct_err"))).first()[0]


def naive_rmspe(df):
    """Persistence baseline: forecast next-window vol as current realized_vol_book.

    Anchor for every model's RMSPE - without it a bare RMSPE number is
    meaningless, since it's not clear whether the model beat "predict no
    change" at all.
    """
    return rmspe(df, pred_col="realized_vol_book")


def time_id_sq_pct_err(df, pred_col, label_col="target"):
    """Per-time_id mean squared percentage error, ordered by time_id.

    Used as the paired loss series for the Diebold-Mariano test: one value
    per time_id lets us compare two forecasts (e.g. model vs. naive
    baseline) on the same units of "independent" observation.
    """
    per_time = (
        df.withColumn(
            "sq_pct_err", F.pow((F.col(label_col) - F.col(pred_col)) / F.col(label_col), 2)
        )
        .groupBy("time_id")
        .agg(F.avg("sq_pct_err").alias("mean_sq_pct_err"))
        .orderBy("time_id")
        .collect()
    )
    return [row["mean_sq_pct_err"] for row in per_time]


def evaluate(model, df, name, naive_rmspe_orig=None):
    pred = model.transform(df)
    pred = pred.withColumn("prediction_orig", F.exp(F.col("prediction")))

    rmse_log = RegressionEvaluator(
        labelCol="log_target", predictionCol="prediction", metricName="rmse"
    ).evaluate(pred)
    r2_log = RegressionEvaluator(
        labelCol="log_target", predictionCol="prediction", metricName="r2"
    ).evaluate(pred)
    rmspe_orig = rmspe(pred)

    scores = {"rmse_log": rmse_log, "r2_log": r2_log, "rmspe_orig": rmspe_orig}

    msg = f"  [{name}] RMSE(log)={rmse_log:.4f}  R2(log)={r2_log:.4f}  RMSPE(orig)={rmspe_orig:.4f}"
    if naive_rmspe_orig is not None:
        pct_improvement = (naive_rmspe_orig - rmspe_orig) / naive_rmspe_orig
        scores["pct_improvement_vs_naive"] = pct_improvement
        msg += f"  ({pct_improvement:+.1%} vs. naive persistence)"
    print(msg)
    return scores


def run():
    spark = get_spark()

    train = load_split(spark, "train").cache()
    val = load_split(spark, "val").cache()
    test = load_split(spark, "test").cache()

    print(f"train={train.count():,}  val={val.count():,}  test={test.count():,}\n")

    naive_val_rmspe = naive_rmspe(val)
    naive_test_rmspe = naive_rmspe(test)
    print(f"Naive baseline (persistence): val RMSPE={naive_val_rmspe:.4f}  test RMSPE={naive_test_rmspe:.4f}\n")

    candidates = {
        "LinearRegression": LinearRegression(
            featuresCol="features", labelCol="log_target", regParam=0.01, elasticNetParam=0.5
        ),
        "GBTRegressor": GBTRegressor(
            featuresCol="features", labelCol="log_target", maxDepth=5, maxIter=50, seed=42
        ),
    }

    fitted = {}
    val_scores = {}
    for name, regressor in candidates.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(regressor)
        model = pipeline.fit(train)
        fitted[name] = model
        print(f"  train metrics:")
        evaluate(model, train, "train")
        print(f"  val metrics:")
        val_scores[name] = evaluate(model, val, "val", naive_rmspe_orig=naive_val_rmspe)
        print()

    best_name = min(val_scores, key=lambda n: val_scores[n]["rmspe_orig"])
    print(f"Best model on validation RMSPE: {best_name}\n")

    print(f"Final test-set evaluation for {best_name}:")
    test_scores = evaluate(fitted[best_name], test, "test", naive_rmspe_orig=naive_test_rmspe)

    best_test_pred = fitted[best_name].transform(test).withColumn(
        "prediction_orig", F.exp(F.col("prediction"))
    )
    model_loss = time_id_sq_pct_err(best_test_pred, pred_col="prediction_orig")
    naive_loss = time_id_sq_pct_err(test, pred_col="realized_vol_book")
    dm_stat, dm_pvalue = diebold_mariano(model_loss, naive_loss)
    significance = "significant (p<0.05)" if dm_pvalue < 0.05 else "not significant (p>=0.05)"
    print(
        f"Diebold-Mariano test ({best_name} vs. naive persistence): "
        f"DM={dm_stat:.4f}  p={dm_pvalue:.4f}  -> {significance}\n"
    )
    test_scores["dm_stat"] = dm_stat
    test_scores["dm_pvalue"] = dm_pvalue

    ensure_dir(MODELS_DIR)
    out_path = MODELS_DIR / "regression_best"
    fitted[best_name].write().overwrite().save(str(out_path))
    print(f"\nsaved best model ({best_name}) to {out_path}")

    final_stage = fitted[best_name].stages[-1]
    importances = None
    if hasattr(final_stage, "featureImportances"):
        importances = {
            col: float(val)
            for col, val in sorted(
                zip(FEATURE_COLS, final_stage.featureImportances.toArray()),
                key=lambda x: -x[1],
            )
        }

    write_results(
        "regression_results",
        {
            "best_model": best_name,
            "naive_baseline": {"val_rmspe": float(naive_val_rmspe), "test_rmspe": float(naive_test_rmspe)},
            "val_scores": {name: {k: float(v) for k, v in s.items()} for name, s in val_scores.items()},
            "test_scores": {k: float(v) for k, v in test_scores.items()},
            "feature_importances": importances,
        },
    )

    spark.stop()
    return best_name, test_scores


if __name__ == "__main__":
    run()
