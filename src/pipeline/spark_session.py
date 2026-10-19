import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "distributed-market-analytics", quiet: bool = False) -> SparkSession:
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

    # quiet=True is for the results notebooks: the stage progress bars
    # ("[Stage 44:====>...]") are decided at session construction, not
    # settable afterwards, so this has to be a builder config rather than a
    # post-hoc setLogLevel call. Pipeline scripts leave this off so job logs
    # stay visible in the Dataproc console.
    if quiet:
        builder = builder.config("spark.ui.showConsoleProgress", "false")

    spark = builder.getOrCreate()
    if quiet:
        spark.sparkContext.setLogLevel("ERROR")
    return spark
