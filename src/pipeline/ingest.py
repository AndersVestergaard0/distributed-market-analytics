"""Ingestion + data-quality cleansing for the full Optiver dataset (112 stocks).

Loads all per-stock partitions of book_train / trade_train and the
(stock_id, time_id) -> target labels, validates schema, drops invalid /
duplicate rows, and writes cleaned Parquet to data/processed/ partitioned by
stock_id so downstream feature engineering can read a single clean source.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_session import get_spark
from paths import DATA_RAW, DATA_PROCESSED, ensure_dir


def load_book(spark: SparkSession) -> DataFrame:
    return spark.read.parquet(str(DATA_RAW / "book_train.parquet"))


def load_trade(spark: SparkSession) -> DataFrame:
    return spark.read.parquet(str(DATA_RAW / "trade_train.parquet"))


def load_labels(spark: SparkSession) -> DataFrame:
    return spark.read.csv(str(DATA_RAW / "train.csv"), header=True, inferSchema=True)


def report_nulls(df: DataFrame, name: str) -> None:
    total = df.count()
    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    nonzero = {k: v for k, v in null_counts.items() if v}
    print(f"[{name}] rows={total:,}  null columns={nonzero or 'none'}")


def clean_book(df: DataFrame) -> DataFrame:
    before = df.count()

    df = df.dropDuplicates(["stock_id", "time_id", "seconds_in_bucket"])

    price_cols = ["bid_price1", "ask_price1", "bid_price2", "ask_price2"]
    size_cols = ["bid_size1", "ask_size1", "bid_size2", "ask_size2"]

    df = df.dropna(subset=price_cols + size_cols + ["time_id", "seconds_in_bucket"])

    # sanity filters: prices must be positive, sizes non-negative, book must
    # not be crossed (best bid below best ask), seconds_in_bucket in [0, 600)
    df = df.filter(
        (F.col("bid_price1") > 0)
        & (F.col("ask_price1") > 0)
        & (F.col("bid_price1") <= F.col("ask_price1"))
        & (F.col("bid_size1") >= 0)
        & (F.col("ask_size1") >= 0)
        & (F.col("seconds_in_bucket") >= 0)
        & (F.col("seconds_in_bucket") < 600)
    )

    after = df.count()
    print(f"[book] cleaned {before:,} -> {after:,} rows ({before - after:,} dropped)")
    return df


def clean_trade(df: DataFrame) -> DataFrame:
    before = df.count()

    df = df.dropDuplicates(["stock_id", "time_id", "seconds_in_bucket"])
    df = df.dropna(subset=["price", "size", "time_id", "seconds_in_bucket"])

    df = df.filter(
        (F.col("price") > 0)
        & (F.col("size") > 0)
        & (F.col("seconds_in_bucket") >= 0)
        & (F.col("seconds_in_bucket") < 600)
    )

    after = df.count()
    print(f"[trade] cleaned {before:,} -> {after:,} rows ({before - after:,} dropped)")
    return df


def clean_labels(df: DataFrame) -> DataFrame:
    before = df.count()
    df = df.dropna().filter(F.col("target") > 0)
    after = df.count()
    print(f"[labels] cleaned {before:,} -> {after:,} rows ({before - after:,} dropped)")
    return df


def run(write_output: bool = True):
    spark = get_spark()

    book = load_book(spark)
    trade = load_trade(spark)
    labels = load_labels(spark)

    print(f"stocks in book data: {book.select('stock_id').distinct().count()}")
    print(f"stocks in trade data: {trade.select('stock_id').distinct().count()}")

    report_nulls(book, "book_train (raw)")
    report_nulls(trade, "trade_train (raw)")
    report_nulls(labels, "train.csv (raw)")

    book_clean = clean_book(book)
    trade_clean = clean_trade(trade)
    labels_clean = clean_labels(labels)

    if write_output:
        ensure_dir(DATA_PROCESSED)
        book_clean.write.mode("overwrite").partitionBy("stock_id").parquet(
            str(DATA_PROCESSED / "book_clean.parquet")
        )
        trade_clean.write.mode("overwrite").partitionBy("stock_id").parquet(
            str(DATA_PROCESSED / "trade_clean.parquet")
        )
        labels_clean.write.mode("overwrite").parquet(
            str(DATA_PROCESSED / "labels_clean.parquet")
        )
        print(f"wrote cleaned data to {DATA_PROCESSED}")

    spark.stop()
    return book_clean, trade_clean, labels_clean


if __name__ == "__main__":
    run()
