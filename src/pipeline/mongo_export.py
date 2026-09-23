"""Writes curated task results to MongoDB (self-hosted via Docker on a GCE
VM, following the same docker-compose setup as INFS3208 Practical 7 - see
GCP_DEPLOYMENT.md).

Each of the four MLlib tasks writes one small, human-readable summary
document (metrics, feature importances, cluster/rule interpretation) rather
than the full prediction tables - MongoDB here is a results/reporting store,
not a second copy of the multi-hundred-million-row raw data.

Set the MONGO_URI environment variable (e.g. "mongodb://<vm-internal-ip>:27017")
to enable this. If it's unset - the default for local development - every
call is a no-op, so the pipeline scripts run identically with or without a
MongoDB instance available.
"""

import os
from datetime import datetime, timezone

DATABASE_NAME = "market_analytics"


def write_results(collection: str, document: dict) -> None:
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        print(f"[mongo_export] MONGO_URI not set - skipping write to '{collection}'")
        return

    from pymongo import MongoClient

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        db = client[DATABASE_NAME]
        document = {**document, "written_at": datetime.now(timezone.utc).isoformat()}
        # One document per task: replace, not append, so each run's results
        # reflect the latest model, not an accumulating history.
        db[collection].replace_one({"_id": collection}, {"_id": collection, **document}, upsert=True)
        print(f"[mongo_export] wrote results to {DATABASE_NAME}.{collection} at {mongo_uri}")
    finally:
        client.close()
