"""Train/validation/test split for the feature table.

time_id is shared across all 112 stocks (each time_id is the same real-world
10-minute window observed simultaneously across stocks), so splitting by row
would leak contemporaneous market-wide information between splits. Instead we
split on the set of distinct time_ids and assign every stock's row for a
given time_id entirely to one split.
"""

from pathlib import Path
import sys

from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402

DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


def run():
    spark = get_spark()
    features = spark.read.parquet(str(DATA_PROCESSED / "features.parquet"))

    time_ids = features.select("time_id").distinct()
    train_ids, val_ids, test_ids = time_ids.randomSplit(
        [TRAIN_FRAC, VAL_FRAC, TEST_FRAC], seed=SEED
    )

    train_ids = train_ids.withColumn("split", F.lit("train"))
    val_ids = val_ids.withColumn("split", F.lit("val"))
    test_ids = test_ids.withColumn("split", F.lit("test"))

    split_map = train_ids.union(val_ids).union(test_ids)

    labelled = features.join(split_map, on="time_id", how="inner")

    counts = labelled.groupBy("split").count().collect()
    counts = {r["split"]: r["count"] for r in counts}
    total = sum(counts.values())
    print("Split sizes (rows):")
    for name in ["train", "val", "test"]:
        n = counts.get(name, 0)
        print(f"  {name}: {n:,} ({n / total:.1%})")

    n_time_ids = time_ids.count()
    print(f"\nDistinct time_ids: {n_time_ids:,}")

    # sanity check: every stock should be represented in every split
    stocks_per_split = (
        labelled.groupBy("split")
        .agg(F.countDistinct("stock_id").alias("n_stocks"))
        .collect()
    )
    print("Stocks represented per split:", {r["split"]: r["n_stocks"] for r in stocks_per_split})

    for name in ["train", "val", "test"]:
        out = labelled.filter(F.col("split") == name).drop("split")
        out.write.mode("overwrite").partitionBy("stock_id").parquet(
            str(DATA_PROCESSED / f"{name}.parquet")
        )
        print(f"wrote {name}.parquet")

    spark.stop()


if __name__ == "__main__":
    run()
