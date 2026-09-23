"""Task 2/4 - Classification: predict trade direction (down/flat/up).

Predicts the direction of the second-half-of-window price move from
first-half-of-window microstructure features (see
src/features/direction_labels.py for why this labeling scheme was used
instead of a "next window" label). Trains a Logistic Regression baseline and
a Random Forest classifier, selects the better model on validation, then
reports a single test-set evaluation.

Metrics: accuracy and weighted F1 (appropriate here since classes are
roughly balanced by construction -- tertile thresholds fit on train).
"""

from pathlib import Path
import sys

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402
from paths import DATA_PROCESSED, MODELS_DIR, ensure_dir  # noqa: E402
from mongo_export import write_results  # noqa: E402

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
    return spark.read.parquet(str(DATA_PROCESSED / "direction_features.parquet")).filter(
        F.col("split") == name
    )


def build_pipeline(classifier):
    indexer = StringIndexer(inputCol="direction_label", outputCol="label")
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    return Pipeline(stages=[indexer, assembler, scaler, classifier])


def evaluate(model, df, name):
    pred = model.transform(df)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(pred)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(pred)

    print(f"  [{name}] accuracy={accuracy:.4f}  weighted_F1={f1:.4f}")
    return {"accuracy": accuracy, "f1": f1}, pred


def confusion_matrix(model, pred):
    labels = model.stages[0].labels  # StringIndexer label order
    to_label = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=labels)
    pred = to_label.transform(pred)

    rows = (
        pred.groupBy("direction_label")
        .pivot("predicted_label", labels)
        .count()
        .fillna(0)
        .orderBy("direction_label")
        .collect()
    )
    return {r["direction_label"]: {c: r[c] for c in labels} for r in rows}


def run():
    spark = get_spark()

    train = load_split(spark, "train").cache()
    val = load_split(spark, "val").cache()
    test = load_split(spark, "test").cache()

    print(f"train={train.count():,}  val={val.count():,}  test={test.count():,}\n")

    candidates = {
        "LogisticRegression": LogisticRegression(
            featuresCol="features", labelCol="label", family="multinomial", regParam=0.01
        ),
        "RandomForestClassifier": RandomForestClassifier(
            featuresCol="features", labelCol="label", numTrees=100, maxDepth=8, seed=42
        ),
    }

    fitted = {}
    val_scores = {}
    for name, clf in candidates.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(clf)
        model = pipeline.fit(train)
        fitted[name] = model
        print("  train metrics:")
        evaluate(model, train, "train")
        print("  val metrics:")
        val_scores[name], _ = evaluate(model, val, "val")
        print()

    best_name = max(val_scores, key=lambda n: val_scores[n]["f1"])
    print(f"Best model on validation weighted F1: {best_name}\n")

    print(f"Final test-set evaluation for {best_name}:")
    test_scores, test_pred = evaluate(fitted[best_name], test, "test")
    cm = confusion_matrix(fitted[best_name], test_pred)
    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    for actual, row in cm.items():
        print(f"    {actual}: {row}")

    ensure_dir(MODELS_DIR)
    out_path = MODELS_DIR / "classification_best"
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
        "classification_results",
        {
            "best_model": best_name,
            "val_scores": {name: {k: float(v) for k, v in s.items()} for name, s in val_scores.items()},
            "test_scores": {k: float(v) for k, v in test_scores.items()},
            "confusion_matrix": cm,
            "feature_importances": importances,
        },
    )

    spark.stop()
    return best_name, test_scores


if __name__ == "__main__":
    run()
