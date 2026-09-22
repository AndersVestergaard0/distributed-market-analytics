from pyspark.sql import SparkSession


def get_spark(app_name: str = "distributed-market-analytics") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[2]")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )
