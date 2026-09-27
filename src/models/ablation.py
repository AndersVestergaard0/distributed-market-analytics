"""Feature ablation for the regression task.

The project's narrative claims OFI and spread/micro-price features matter
for forecasting volatility - this re-fits the winning architecture (GBT)
with each feature group removed and reports the RMSPE delta vs. the full
feature set, to back that claim with a number.
"""

from pathlib import Path
import sys

from pyspark.ml.regression import GBTRegressor

sys.path.append(str(Path(__file__).resolve().parent))
from regression import FEATURE_COLS, build_pipeline, evaluate, load_split  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402
from mongo_export import write_results  # noqa: E402

ABLATIONS = {
    "without_ofi": ["total_ofi", "avg_ofi"],
    "without_spread_microprice": ["avg_spread_bps", "avg_micro_price"],
}


def gbt_regressor():
    """Matches the GBTRegressor hyperparameters used in regression.py."""
    return GBTRegressor(featuresCol="features", labelCol="log_target", maxDepth=5, maxIter=50, seed=42)


def run():
    spark = get_spark()

    train = load_split(spark, "train").cache()
    val = load_split(spark, "val").cache()
    test = load_split(spark, "test").cache()

    print(f"train={train.count():,}  val={val.count():,}  test={test.count():,}\n")

    print("Full feature set:")
    full_pipeline = build_pipeline(gbt_regressor())
    full_model = full_pipeline.fit(train)
    full_val_scores = evaluate(full_model, val, "val")
    full_test_scores = evaluate(full_model, test, "test")
    print()

    results = {
        "full": {
            "removed_features": [],
            "val_scores": {k: float(v) for k, v in full_val_scores.items()},
            "test_scores": {k: float(v) for k, v in full_test_scores.items()},
        }
    }

    for ablation_name, drop_cols in ABLATIONS.items():
        reduced_cols = [c for c in FEATURE_COLS if c not in drop_cols]
        print(f"Ablation '{ablation_name}' (dropping {drop_cols}):")

        pipeline = build_pipeline(gbt_regressor())
        # build_pipeline() closes over FEATURE_COLS via VectorAssembler's
        # inputCols default, so override it explicitly for this ablation's
        # reduced column set rather than mutating the module-level constant.
        pipeline.getStages()[0].setInputCols(reduced_cols)

        model = pipeline.fit(train)
        val_scores = evaluate(model, val, "val")
        test_scores = evaluate(model, test, "test")

        rmspe_delta_val = val_scores["rmspe_orig"] - full_val_scores["rmspe_orig"]
        rmspe_delta_test = test_scores["rmspe_orig"] - full_test_scores["rmspe_orig"]
        print(
            f"  RMSPE delta vs. full features: val={rmspe_delta_val:+.4f}  "
            f"test={rmspe_delta_test:+.4f}\n"
        )

        results[ablation_name] = {
            "removed_features": drop_cols,
            "val_scores": {k: float(v) for k, v in val_scores.items()},
            "test_scores": {k: float(v) for k, v in test_scores.items()},
            "rmspe_delta_vs_full": {"val": float(rmspe_delta_val), "test": float(rmspe_delta_test)},
        }

    write_results("ablation_results", results)

    spark.stop()
    return results


if __name__ == "__main__":
    run()
