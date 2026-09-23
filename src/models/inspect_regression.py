from pathlib import Path
import sys

from pyspark.ml import PipelineModel

sys.path.append(str(Path(__file__).resolve().parents[1] / "pipeline"))
from spark_session import get_spark  # noqa: E402
from regression import FEATURE_COLS, MODELS_DIR  # noqa: E402

spark = get_spark()
model = PipelineModel.load(str(MODELS_DIR / "regression_best"))
gbt = model.stages[-1]

importances = list(zip(FEATURE_COLS, gbt.featureImportances.toArray()))
importances.sort(key=lambda x: -x[1])

print("Feature importances (GBTRegressor):")
for name, val in importances:
    print(f"  {name:<20s} {val:.4f}")

print(f"\nNumber of trees: {gbt.getNumTrees}")
print(f"Max depth: {gbt.getOrDefault('maxDepth')}")

spark.stop()
