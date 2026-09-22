"""Task 3/4 - Clustering: volatility regime grouping (K-Means).

The EDA showed stocks have structurally different baseline volatility, so
clustering on raw (globally-scaled) features risks clusters that mostly
rediscover "which stock is this" rather than a genuine cross-stock market
regime. Instead, each feature is z-scored *relative to its own stock's*
mean/std (fit on train only, applied unchanged to val/test - same
no-leakage discipline used for the classification thresholds), so a cluster
means "this window was unusually calm/volatile *for this stock*", which is
comparable across different stocks.

k is selected by silhouette score on the validation set over a small range.
"""

from pathlib import Path
import sys

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402
from regression import FEATURE_COLS, DATA_PROCESSED, load_split  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

K_CANDIDATES = [2, 3, 4, 5]


def fit_per_stock_stats(train):
    """Per-stock (mean, stddev) for each feature, computed on train only."""
    aggs = []
    for c in FEATURE_COLS:
        aggs.append(F.avg(c).alias(f"{c}__mean"))
        aggs.append(F.stddev(c).alias(f"{c}__std"))
    return train.groupBy("stock_id").agg(*aggs)


def apply_per_stock_zscore(df, stats):
    df = df.join(stats, on="stock_id", how="inner")
    for c in FEATURE_COLS:
        mean_col, std_col = f"{c}__mean", f"{c}__std"
        df = df.withColumn(
            f"z_{c}",
            F.when(F.col(std_col) > 0, (F.col(c) - F.col(mean_col)) / F.col(std_col)).otherwise(0.0),
        )
    return df


def build_pipeline(k):
    z_cols = [f"z_{c}" for c in FEATURE_COLS]
    assembler = VectorAssembler(inputCols=z_cols, outputCol="features")
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=42)
    return Pipeline(stages=[assembler, kmeans])


def silhouette(model, df):
    pred = model.transform(df)
    return ClusteringEvaluator(featuresCol="features", predictionCol="cluster").evaluate(pred), pred


def describe_clusters(pred):
    """Cluster interpretation in original (non z-scored) units."""
    describe_cols = ["realized_vol_book", "avg_spread_bps", "trade_volume", "target"]
    return (
        pred.groupBy("cluster")
        .agg(
            F.count("*").alias("n_windows"),
            *[F.avg(c).alias(f"avg_{c}") for c in describe_cols],
        )
        .orderBy("cluster")
    )


def run():
    spark = get_spark()

    train = load_split(spark, "train").cache()
    val = load_split(spark, "val").cache()
    test = load_split(spark, "test").cache()

    stats = fit_per_stock_stats(train).cache()

    train_z = apply_per_stock_zscore(train, stats)
    val_z = apply_per_stock_zscore(val, stats)
    test_z = apply_per_stock_zscore(test, stats)

    print(f"train={train.count():,}  val={val.count():,}  test={test.count():,}\n")

    best_k, best_model, best_val_sil = None, None, -1
    for k in K_CANDIDATES:
        pipeline = build_pipeline(k)
        model = pipeline.fit(train_z)
        val_sil, _ = silhouette(model, val_z)
        print(f"k={k}: validation silhouette = {val_sil:.4f}")
        if val_sil > best_val_sil:
            best_k, best_model, best_val_sil = k, model, val_sil

    print(f"\nBest k on validation silhouette: {best_k} (silhouette={best_val_sil:.4f})\n")

    test_sil, test_pred = silhouette(best_model, test_z)
    print(f"Test silhouette (k={best_k}): {test_sil:.4f}\n")

    print("Cluster interpretation (test set, original units):")
    describe_clusters(test_pred).show(truncate=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "clustering_best"
    best_model.write().overwrite().save(str(out_path))
    stats.write.mode("overwrite").parquet(str(DATA_PROCESSED / "cluster_stock_stats.parquet"))
    print(f"saved best model (k={best_k}) to {out_path}")

    spark.stop()
    return best_k, test_sil


if __name__ == "__main__":
    run()
