import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "distributed-market-analytics") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.driver.maxResultSize", "1g")
    )

    # Only force local mode for local dev runs against data/. On Dataproc
    # (DATA_ROOT set to a gs:// URI) let spark-submit's own --master (yarn)
    # stand, so jobs actually use the cluster's worker nodes instead of being
    # confined to a single 2-thread local JVM on the driver.
    if not os.environ.get("DATA_ROOT", "").startswith("gs://"):
        builder = builder.master("local[2]").config("spark.driver.memory", "3g")

    return builder.getOrCreate()
