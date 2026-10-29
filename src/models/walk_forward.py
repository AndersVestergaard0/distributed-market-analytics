"""Fold-stability check for the regression task.

Instead of one static train/val/test split, trains on every time_id in folds
before fold i and evaluates on fold i, for each fold produced by
split.write_time_id_folds(). time_id is anonymised and not chronologically
ordered, so the fold order is an arbitrary but fixed partition of the data,
not real chronology - this is NOT true walk-forward evaluation and makes no
past-vs-future guarantee. It is still a useful stability check: it shows
whether RMSPE holds up across several non-overlapping subsets of the data,
rather than relying on a single train/val/test split that could be a lucky
cut, as in regression.py.
"""

from pathlib import Path
import statistics
import sys

from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parent))
from ablation import gbt_regressor  # noqa: E402
from regression import build_pipeline, evaluate  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402
from paths import DATA_PROCESSED  # noqa: E402
from mongo_export import write_results  # noqa: E402


def load_labelled_features(spark):
    features = spark.read.parquet(str(DATA_PROCESSED / "features.parquet"))
    folds = spark.read.parquet(str(DATA_PROCESSED / "time_id_folds.parquet"))
    labelled = features.join(folds, on="time_id", how="inner")
    return labelled.withColumn("log_target", F.log(F.col("target")))


def run():
    spark = get_spark()
    labelled = load_labelled_features(spark).cache()

    n_folds = labelled.select("fold").distinct().count()
    print(f"Walk-forward evaluation across {n_folds} folds\n")

    fold_scores = []
    for fold in range(1, n_folds):
        train = labelled.filter(F.col("fold") < fold)
        test = labelled.filter(F.col("fold") == fold)

        train_n, test_n = train.count(), test.count()
        print(f"Fold {fold}: train={train_n:,} rows (folds 0..{fold - 1})  test={test_n:,} rows (fold {fold})")

        pipeline = build_pipeline(gbt_regressor())
        model = pipeline.fit(train)
        scores = evaluate(model, test, f"fold {fold}")
        fold_scores.append(scores["rmspe_orig"])
        print()

    mean_rmspe = statistics.mean(fold_scores)
    std_rmspe = statistics.stdev(fold_scores) if len(fold_scores) > 1 else 0.0
    print(f"RMSPE across folds: mean={mean_rmspe:.4f}  std={std_rmspe:.4f}")
    print(f"Per-fold RMSPE: {[round(s, 4) for s in fold_scores]}")

    write_results(
        "walk_forward_results",
        {
            "n_folds": n_folds,
            "per_fold_rmspe": [float(s) for s in fold_scores],
            "mean_rmspe": float(mean_rmspe),
            "std_rmspe": float(std_rmspe),
        },
    )

    spark.stop()
    return fold_scores


if __name__ == "__main__":
    run()
