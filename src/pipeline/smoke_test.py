"""Quick sanity check: load a couple of stocks' order book + trade data and
print schema/row counts to confirm the local Spark setup can read the
Optiver dataset before any real pipeline logic is written."""

from pathlib import Path

from spark_session import get_spark

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def main():
    spark = get_spark()

    book = spark.read.parquet(str(DATA_RAW / "book_train.parquet" / "stock_id=0"))
    trade = spark.read.parquet(str(DATA_RAW / "trade_train.parquet" / "stock_id=0"))
    train = spark.read.csv(str(DATA_RAW / "train.csv"), header=True, inferSchema=True)

    print("\n--- book_train (stock_id=0) schema ---")
    book.printSchema()
    print("row count:", book.count())

    print("\n--- trade_train (stock_id=0) schema ---")
    trade.printSchema()
    print("row count:", trade.count())

    print("\n--- train.csv schema ---")
    train.printSchema()
    print("row count:", train.count())

    spark.stop()


if __name__ == "__main__":
    main()
