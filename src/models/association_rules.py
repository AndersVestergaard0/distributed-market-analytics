"""Task 4/4 - Association Rule Mining: what conditions co-occur with a
high-volatility next window (FP-Growth).

Each engineered feature is discretized into low/mid/high tertile bins - the
tertile edges are fit on the train split only (same no-leakage discipline
used for the classification labels) and applied unchanged to val/test. Each
window becomes a "transaction": a set of items like
{"avg_spread_bps=high", "realized_vol_book=high", "target=high"}. FP-Growth
then mines frequent itemsets and association rules over these transactions.

This is descriptive/unsupervised, not a predictive model with a train/test
accuracy figure - but to demonstrate the mined rules aren't just an artifact
of the train split, the top rules whose consequent is "target=high" (i.e.
rules that flag an upcoming high-volatility window) are re-checked against
the test split: their support and confidence are recomputed there and
compared to what FP-Growth found on train.
"""

from pathlib import Path
import sys

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402
from regression import DATA_PROCESSED, load_split  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

# Subset of engineered features to mine over, plus the prediction target
# itself (binned) so rules can surface conditions associated with an
# upcoming high-volatility window.
MINE_COLS = [
    "avg_spread_bps",
    "total_ofi",
    "trade_volume",
    "trade_count",
    "realized_vol_book",
]
OUTCOME_COL = "target"
ALL_COLS = MINE_COLS + [OUTCOME_COL]

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.3


def fit_tertile_edges(train, col):
    q1, q2 = train.approxQuantile(col, [1 / 3, 2 / 3], 0.01)
    return q1, q2


def add_item_col(df, col, q1, q2):
    bin_expr = (
        F.when(F.col(col) < q1, "low")
        .when(F.col(col) < q2, "mid")
        .otherwise("high")
    )
    return df.withColumn(f"{col}__item", F.concat(F.lit(f"{col}="), bin_expr))


def add_all_items(df, edges):
    for col in ALL_COLS:
        df = add_item_col(df, col, *edges[col])
    return df.withColumn("items", F.array(*[f"{c}__item" for c in ALL_COLS]))


def run():
    spark = get_spark()

    train = load_split(spark, "train").cache()
    val = load_split(spark, "val").cache()
    test = load_split(spark, "test").cache()

    print(f"train={train.count():,}  val={val.count():,}  test={test.count():,}\n")

    edges = {c: fit_tertile_edges(train, c) for c in ALL_COLS}
    print("Tertile edges (fit on train only):")
    for c, (q1, q2) in edges.items():
        print(f"  {c}: low<{q1:.4g}<=mid<{q2:.4g}<=high")
    print()

    train_items = add_all_items(train, edges).select("items").cache()
    val_items = add_all_items(val, edges).select("items")
    test_items = add_all_items(test, edges).select("items").cache()

    fpgrowth = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE)
    model = fpgrowth.fit(train_items)

    print(f"Frequent itemsets (train, minSupport={MIN_SUPPORT}): "
          f"{model.freqItemsets.count()} itemsets found")
    print("Top 10 by frequency:")
    model.freqItemsets.orderBy(F.desc("freq")).show(10, truncate=False)

    rules = model.associationRules
    print(f"\nAssociation rules (train, minConfidence={MIN_CONFIDENCE}): "
          f"{rules.count()} rules found")

    # Rules that flag an upcoming high-volatility window.
    vol_rules = (
        rules.filter(F.array_contains(F.col("consequent"), "target=high"))
        .orderBy(F.desc("confidence"))
    )
    print("\nRules predicting target=high (upcoming high-volatility window), "
          "sorted by confidence:")
    vol_rules.show(20, truncate=False)

    top_rules = vol_rules.limit(5).collect()

    print("\nValidating top rules on the held-out test split:")
    test_n = test_items.count()
    for r in top_rules:
        antecedent = list(r["antecedent"])
        cond = None
        for item in antecedent:
            c = F.array_contains(F.col("items"), item)
            cond = c if cond is None else (cond & c)
        support_n = test_items.filter(cond).count()
        confident_n = test_items.filter(cond & F.array_contains(F.col("items"), "target=high")).count()
        test_support = support_n / test_n
        test_confidence = confident_n / support_n if support_n > 0 else float("nan")
        print(
            f"  {antecedent} => [target=high] | "
            f"train: support={r['support']:.4f} confidence={r['confidence']:.4f} lift={r['lift']:.4f} | "
            f"test: support={test_support:.4f} confidence={test_confidence:.4f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "association_rules_best"
    model.write().overwrite().save(str(out_path))
    rules.write.mode("overwrite").parquet(str(DATA_PROCESSED / "association_rules.parquet"))
    print(f"\nsaved model to {out_path}")

    spark.stop()


if __name__ == "__main__":
    run()
