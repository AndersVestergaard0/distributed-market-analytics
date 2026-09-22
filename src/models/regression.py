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

DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

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


def evaluate(model, df, name):
    pred = model.transform(df)
    pred = pred.withColumn("prediction_orig", F.exp(F.col("prediction")))

    rmse_log = RegressionEvaluator(
        labelCol="log_target", predictionCol="prediction", metricName="rmse"
    ).evaluate(pred)
    r2_log = RegressionEvaluator(
        labelCol="log_target", predictionCol="prediction", metricName="r2"
    ).evaluate(pred)
    rmspe_orig = rmspe(pred)

    print(f"  [{name}] RMSE(log)={rmse_log:.4f}  R2(log)={r2_log:.4f}  RMSPE(orig)={rmspe_orig:.4f}")
    return {"rmse_log": rmse_log, "r2_log": r2_log, "rmspe_orig": rmspe_orig}


def run():
    spark = get_spark()

    train = load_split(spark, "train").cache()
    val = load_split(spark, "val").cache()
    test = load_split(spark, "test").cache()

    print(f"train={train.count():,}  val={val.count():,}  test={test.count():,}\n")

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
        val_scores[name] = evaluate(model, val, "val")
        print()

    best_name = min(val_scores, key=lambda n: val_scores[n]["rmspe_orig"])
    print(f"Best model on validation RMSPE: {best_name}\n")

    print(f"Final test-set evaluation for {best_name}:")
    test_scores = evaluate(fitted[best_name], test, "test")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "regression_best"
    fitted[best_name].write().overwrite().save(str(out_path))
    print(f"\nsaved best model ({best_name}) to {out_path}")

    spark.stop()
    return best_name, test_scores


if __name__ == "__main__":
    run()
