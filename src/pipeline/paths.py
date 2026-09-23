"""Resolves data/model storage roots to either the local filesystem or GCS.

Set the DATA_ROOT environment variable to a gs://<bucket>/... URI (passed via
`--properties spark.executorEnv.DATA_ROOT=...` on a Dataproc job, or exported
in the shell) to run the exact same scripts against Google Cloud Storage
instead of the local data/ directory - no other code changes needed.
"""

import os
from pathlib import Path


class DataPath(str):
    """Drop-in for pathlib's `/` joining that also works with gs:// URIs -
    plain pathlib.Path collapses "gs://bucket" down to "gs:/bucket"."""

    def __truediv__(self, other):
        return DataPath(f"{str(self).rstrip('/')}/{other}")


def _root() -> str:
    root = os.environ.get("DATA_ROOT")
    if root:
        return root.rstrip("/")
    return str(Path(__file__).resolve().parents[2] / "data")


def ensure_dir(path) -> None:
    """mkdir -p; a no-op for gs:// paths, which have no directories to create."""
    if not str(path).startswith("gs://"):
        Path(str(path)).mkdir(parents=True, exist_ok=True)


DATA_ROOT = DataPath(_root())
DATA_RAW = DATA_ROOT / "raw"
DATA_PROCESSED = DATA_ROOT / "processed"
MODELS_DIR = DATA_ROOT / "models"
