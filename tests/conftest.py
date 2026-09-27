"""Shared pytest fixtures: a local Spark session and sys.path wiring for the
src/ modules, which import each other by bare module name (no src/__init__.py
package structure) the same way the pipeline scripts do."""

from pathlib import Path
import os
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ["features", "models", "pipeline"]:
    sys.path.append(str(ROOT / "src" / sub))

# PySpark workers otherwise spawn whatever `python3` is first on PATH, which
# can be a different minor version than the interpreter running pytest (this
# venv) and PySpark refuses to run with mismatched driver/worker versions.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from spark_session import get_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    session = get_spark(app_name="distributed-market-analytics-tests")
    yield session
    session.stop()
