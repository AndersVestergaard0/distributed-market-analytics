"""Market microstructure feature engineering.

Reads the cleaned book/trade/label Parquet written by src/pipeline/ingest.py
and computes, per (stock_id, time_id) 10-minute window:

- micro-price, bid-ask spread (book-level, per snapshot then aggregated)
- Order Flow Imbalance (OFI) between consecutive book snapshots
- rolling realized volatility of log mid-price returns
- trade-side aggregates: total volume, trade count, avg trade size

Output is one row per (stock_id, time_id), joined with the target label,
ready to feed the four Spark MLlib tasks.
"""

from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402

DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def add_price_features(book: DataFrame) -> DataFrame:
    """Per-snapshot micro-price, mid-price, spread. Cheap: no shuffle/window."""

    return book.withColumn(
        "mid_price", (F.col("bid_price1") + F.col("ask_price1")) / 2
    ).withColumn(
        "micro_price",
        (
            F.col("bid_price1") * F.col("ask_size1")
            + F.col("ask_price1") * F.col("bid_size1")
        )
        / (F.col("bid_size1") + F.col("ask_size1")),
    ).withColumn(
        "spread", F.col("ask_price1") - F.col("bid_price1")
    ).withColumn(
        "spread_bps", F.col("spread") / F.col("mid_price") * 10000
    )


def add_flow_features(book: DataFrame) -> DataFrame:
    """Order Flow Imbalance + log returns between consecutive snapshots.

    Expensive: requires a partitioned/ordered window (shuffle + sort). Call
    this only on the subset of rows actually needed (e.g. the early half of
    a window for the classification task), not on data that will be
    discarded afterwards.
    """

    # Order Flow Imbalance: change in bid size minus change in ask size
    # between consecutive snapshots within the same (stock_id, time_id).
    w = Window.partitionBy("stock_id", "time_id").orderBy("seconds_in_bucket")

    book = book.withColumn("prev_bid_size1", F.lag("bid_size1").over(w))
    book = book.withColumn("prev_ask_size1", F.lag("ask_size1").over(w))
    book = book.withColumn("prev_bid_price1", F.lag("bid_price1").over(w))
    book = book.withColumn("prev_ask_price1", F.lag("ask_price1").over(w))

    bid_flow = F.when(
        F.col("bid_price1") > F.col("prev_bid_price1"), F.col("bid_size1")
    ).when(
        F.col("bid_price1") == F.col("prev_bid_price1"),
        F.col("bid_size1") - F.col("prev_bid_size1"),
    ).otherwise(-F.col("prev_bid_size1"))

    ask_flow = F.when(
        F.col("ask_price1") < F.col("prev_ask_price1"), F.col("ask_size1")
    ).when(
        F.col("ask_price1") == F.col("prev_ask_price1"),
        F.col("ask_size1") - F.col("prev_ask_size1"),
    ).otherwise(-F.col("prev_ask_size1"))

    book = book.withColumn("ofi", F.coalesce(bid_flow - ask_flow, F.lit(0.0)))

    # log return of mid-price between consecutive snapshots, for realized vol
    book = book.withColumn(
        "log_return",
        F.when(
            F.col("prev_bid_price1").isNotNull(),
            F.log(F.col("mid_price") / ((F.col("prev_bid_price1") + F.col("prev_ask_price1")) / 2)),
        ),
    )

    return book


def add_book_snapshot_features(book: DataFrame) -> DataFrame:
    """Full per-snapshot feature set (price + flow). Used by the regression
    task, which needs OFI/realized-vol computed across the whole window."""
    return add_flow_features(add_price_features(book))


def aggregate_book_features(book: DataFrame) -> DataFrame:
    """Collapse per-snapshot book features to one row per (stock_id, time_id)."""

    return book.groupBy("stock_id", "time_id").agg(
        F.avg("spread_bps").alias("avg_spread_bps"),
        F.avg("micro_price").alias("avg_micro_price"),
        F.stddev("mid_price").alias("mid_price_stddev"),
        F.sum("ofi").alias("total_ofi"),
        F.avg("ofi").alias("avg_ofi"),
        # realized volatility: sqrt of sum of squared log returns
        F.sqrt(F.sum(F.pow(F.col("log_return"), 2))).alias("realized_vol_book"),
        F.count("*").alias("book_update_count"),
    )


def aggregate_trade_features(trade: DataFrame) -> DataFrame:
    return trade.groupBy("stock_id", "time_id").agg(
        F.sum("size").alias("trade_volume"),
        F.count("*").alias("trade_count"),
        F.avg("size").alias("avg_trade_size"),
        F.avg("order_count").alias("avg_order_count"),
        F.stddev("price").alias("trade_price_stddev"),
    )


def build_feature_table(spark) -> DataFrame:
    book = spark.read.parquet(str(DATA_PROCESSED / "book_clean.parquet"))
    trade = spark.read.parquet(str(DATA_PROCESSED / "trade_clean.parquet"))
    labels = spark.read.parquet(str(DATA_PROCESSED / "labels_clean.parquet"))

    book = add_book_snapshot_features(book)
    book_features = aggregate_book_features(book)
    trade_features = aggregate_trade_features(trade)

    features = (
        book_features.join(trade_features, on=["stock_id", "time_id"], how="left")
        .join(labels, on=["stock_id", "time_id"], how="inner")
        .fillna(
            {
                "trade_volume": 0,
                "trade_count": 0,
                "avg_trade_size": 0,
                "avg_order_count": 0,
                "trade_price_stddev": 0.0,
            }
        )
    )

    return features


def run():
    spark = get_spark()
    features = build_feature_table(spark)

    print("feature table schema:")
    features.printSchema()
    row_count = features.count()
    print(f"feature table rows: {row_count:,}  (one row per stock_id/time_id window)")

    features.show(5, truncate=False)

    out_path = DATA_PROCESSED / "features.parquet"
    features.write.mode("overwrite").partitionBy("stock_id").parquet(str(out_path))
    print(f"wrote feature table to {out_path}")

    spark.stop()
    return features


if __name__ == "__main__":
    run()
