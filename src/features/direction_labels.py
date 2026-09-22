"""Feature + label table for the trade-direction classification task.

time_id values are anonymised and NOT chronologically ordered (Optiver's own
documentation), so unlike the regression task (which consumes Optiver's own
pre-computed next-window target), we cannot safely build a "next window"
label ourselves.

Instead we use seconds_in_bucket -- a real, reliable clock *within* each
10-minute window -- to build a genuine forecasting split:
  - predictors: engineered from the FIRST half of the window only
    (seconds_in_bucket < 300)
  - label: direction of the mid-price move from the end of the first half
    to the end of the window (seconds_in_bucket >= 300)

Label thresholds (down / flat / up) are tertiles of the log-return
distribution computed on the TRAIN split only, then applied unchanged to
val/test, so classes start roughly balanced and no threshold information
leaks from val/test.
"""

from pathlib import Path
import sys

from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parent))
from microstructure import add_price_features, add_flow_features, aggregate_book_features, aggregate_trade_features  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402

DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

EARLY_CUTOFF = 300  # seconds; window is 0-599


def build_direction_table(spark):
    book = spark.read.parquet(str(DATA_PROCESSED / "book_clean.parquet"))
    trade = spark.read.parquet(str(DATA_PROCESSED / "trade_clean.parquet"))
    split_map = spark.read.parquet(str(DATA_PROCESSED / "time_id_split.parquet"))

    # cheap, non-shuffling price features first, on the full table
    book = add_price_features(book)

    early_book = book.filter(F.col("seconds_in_bucket") < EARLY_CUTOFF)
    late_book = book.filter(F.col("seconds_in_bucket") >= EARLY_CUTOFF)
    early_trade = trade.filter(F.col("seconds_in_bucket") < EARLY_CUTOFF)

    # the expensive windowed OFI/log-return computation only needs to run on
    # the early half -- the late half is only used for its mid_price at the
    # last snapshot (for the label), which needs no window function at all
    early_book = add_flow_features(early_book)

    # predictors: same feature aggregations as the regression task, but
    # computed on the early half of the window only
    early_book_features = aggregate_book_features(early_book)
    early_trade_features = aggregate_trade_features(early_trade)

    # label: mid-price at the end of the early half vs. the end of the window
    early_end = early_book.groupBy("stock_id", "time_id").agg(
        F.max_by("mid_price", "seconds_in_bucket").alias("mid_price_early_end")
    )
    late_end = late_book.groupBy("stock_id", "time_id").agg(
        F.max_by("mid_price", "seconds_in_bucket").alias("mid_price_late_end")
    )

    labels = early_end.join(late_end, on=["stock_id", "time_id"], how="inner").withColumn(
        "log_return_2nd_half",
        F.log(F.col("mid_price_late_end") / F.col("mid_price_early_end")),
    )

    table = (
        early_book_features.join(early_trade_features, on=["stock_id", "time_id"], how="left")
        .join(labels, on=["stock_id", "time_id"], how="inner")
        .join(split_map, on="time_id", how="inner")
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

    return table


def add_tertile_labels(table):
    """Fit down/flat/up thresholds on train only, apply to all splits."""
    train_returns = table.filter(F.col("split") == "train").select("log_return_2nd_half")
    q1, q2 = train_returns.approxQuantile("log_return_2nd_half", [1 / 3, 2 / 3], 0.001)
    print(f"tertile thresholds (fit on train): q1={q1:.6f}  q2={q2:.6f}")

    table = table.withColumn(
        "direction_label",
        F.when(F.col("log_return_2nd_half") <= q1, F.lit("down"))
        .when(F.col("log_return_2nd_half") >= q2, F.lit("up"))
        .otherwise(F.lit("flat")),
    )
    return table, (q1, q2)


def run():
    spark = get_spark()
    table = build_direction_table(spark)
    table, thresholds = add_tertile_labels(table)

    print("\nClass balance per split:")
    table.groupBy("split", "direction_label").count().orderBy("split", "direction_label").show()

    out_path = DATA_PROCESSED / "direction_features.parquet"
    table.write.mode("overwrite").partitionBy("split").parquet(str(out_path))
    print(f"wrote direction feature table to {out_path}")

    spark.stop()
    return table


if __name__ == "__main__":
    run()
