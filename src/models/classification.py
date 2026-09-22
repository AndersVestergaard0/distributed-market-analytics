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


def print_confusion_matrix(model, pred):
    labels = model.stages[0].labels  # StringIndexer label order
    to_label = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=labels)
    pred = to_label.transform(pred)

    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    pred.groupBy("direction_label").pivot("predicted_label", labels).count().orderBy(
        "direction_label"
    ).show()


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
    print_confusion_matrix(fitted[best_name], test_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "classification_best"
    fitted[best_name].write().overwrite().save(str(out_path))
    print(f"\nsaved best model ({best_name}) to {out_path}")

    spark.stop()
    return best_name, test_scores


if __name__ == "__main__":
    run()
