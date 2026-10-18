# GCP Deployment - Setup Commands

Full command reference for deploying this project to Google Cloud:
**GCS** (data lake) → **Dataproc** (Spark compute, run as a one-click
Workflow Template) → **MongoDB on a GCE VM** (results store, same
docker-compose pattern as Practical 7) → **Jupyter via Dataproc Component
Gateway** (visualization).

Run everything below yourself from a terminal with `gcloud` installed and
authenticated (`gcloud init` / `gcloud auth login`). Once the Workflow Template exists (Part 4), the
whole pipeline re-runs from the Console with a single click - no further
CLI needed - which is the point of Part 4.

## Running this from Cloud Shell

Every command below also runs unmodified in **Cloud Shell** (the `>_` icon
in the top-right of the GCP Console) - it ships with `gcloud`, `gsutil`,
`docker`, and `zip` preinstalled and already authenticated as you, so it's
the easiest way to do this setup if you don't have `gcloud` installed
locally:

```bash
git clone https://github.com/AndersVestergaard0/distributed-market-analytics.git
cd distributed-market-analytics
```

Cloud Shell's persistent home directory is capped at 5GB, so don't upload
`data/raw` (2.5GB) from your laptop through the Console's file picker -
instead re-download it straight from Kaggle inside Cloud Shell, the same way
it was first fetched locally:

```bash
# Upload your kaggle.json first via Cloud Shell's Upload button (⋮ menu),
# then:
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
pip install --user kaggle
~/.local/bin/kaggle datasets download akshaymairal/optiver-realized-volatility-prediction -p data/raw --unzip
```

Then continue from Part 1 below. **This download/upload step (Part 1) only
needs to happen once**, done by you - see "Access for your teacher" at the
end of this doc for why they don't need to repeat it.

---

## 0. One-time variables

Set these once per terminal session; every command below reuses them.

```bash
export PROJECT_ID="your-gcp-project-id"  # an existing project, or create one at console.cloud.google.com/projectcreate
export REGION="australia-southeast1"     # any Dataproc-supported region; pick whatever's closest to you
export ZONE="australia-southeast1-b"
export BUCKET="${PROJECT_ID}-market-analytics"
export CLUSTER_JUPYTER="market-analytics-jupyter"
export MONGO_VM="market-analytics-mongo"

# MongoDB auth (Part 2) - pick your own values, then keep them consistent
# across every session (Part 2 provisions the users, Part 3 bakes
# MONGO_WRITER_PASSWORD into the Dataproc jobs' connection string). Only
# MONGO_READER_PASSWORD is even worth calling a "password" here - it
# gates a read-only user that mongo-express itself holds server-side and
# never exposes to a visitor's browser (see Part 2), so mongo-express's port
# is safe to leave open to the public internet.
export MONGO_ROOT_PASSWORD="change-me-root"
export MONGO_WRITER_PASSWORD="change-me-writer"
export MONGO_READER_PASSWORD="change-me-reader"

gcloud config set project "$PROJECT_ID"
gcloud services enable dataproc.googleapis.com compute.googleapis.com storage.googleapis.com
```

---

## 1. GCS - data lake

```bash
gsutil mb -l "$REGION" "gs://$BUCKET"

# raw data only - processed/models are written by the pipeline itself
gsutil -m rsync -r data/raw "gs://$BUCKET/data/raw"

# pipeline code, flattened into one importable zip (see note below)
cd src
zip -j code.zip pipeline/*.py features/*.py models/*.py
gsutil cp code.zip "gs://$BUCKET/code/code.zip"
for f in pipeline/*.py features/*.py models/*.py; do
  gsutil cp "$f" "gs://$BUCKET/code/$(basename "$f")"
done
cd ..

# results notebooks, for browsing via the Jupyter file browser (Part 5)
gsutil -m cp notebooks/*.ipynb "gs://$BUCKET/notebooks/"

# a tiny self-hosted init action (Part 3) - more reliable than depending on
# Google's community initialization-actions repo, whose file paths change
cat > init-pip-install.sh <<'EOF'
#!/bin/bash
pip install pymongo scikit-learn
EOF
gsutil cp init-pip-install.sh "gs://$BUCKET/code/init-pip-install.sh"
```

> **Why `zip -j` (flatten paths)?** Every script imports its dependencies
> unqualified (`from spark_session import get_spark`, `from regression import
> FEATURE_COLS`, ...). Flattening `pipeline/`, `features/`, `models/` into one
> zip root and passing it as `--py-files` on every job submission (Part 3/4)
> puts all of them on `PYTHONPATH` together, so those imports resolve exactly
> as they do locally - no code changes needed for the cloud.

---

## 2. MongoDB - GCE VM (same docker-compose pattern as Practical 7)

The VM boots with [`deploy/mongo-startup.sh`](deploy/mongo-startup.sh), which
installs Docker and brings up `mongo` + `mongo-express` with auth enabled: a
`writer` user (read/write, used by the Dataproc jobs in Part 3) and a
`public_reader` user (read-only, the *only* credential mongo-express itself
holds). Because mongo-express can't do anything but read no matter what a
visitor clicks, its port is safe to open to the public internet - that's
what lets you link the results dashboard straight from the README for your
teacher/reviewers without granting anyone GCP access.

```bash
gcloud compute instances create "$MONGO_VM" \
  --zone="$ZONE" \
  --machine-type=e2-small \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --tags=mongo-vm \
  --metadata=MONGO_ROOT_PASSWORD="$MONGO_ROOT_PASSWORD",MONGO_WRITER_PASSWORD="$MONGO_WRITER_PASSWORD",MONGO_READER_PASSWORD="$MONGO_READER_PASSWORD" \
  --metadata-from-file=startup-script=deploy/mongo-startup.sh

# internal traffic (Dataproc -> Mongo, port 27017) - restrict to your VPC's
# range; this is the only path that can write, and it needs the "writer"
# credential regardless, so it never needs to be public
gcloud compute firewall-rules create allow-mongo-internal \
  --network=default --direction=INGRESS --action=ALLOW \
  --rules=tcp:27017 --source-ranges=10.128.0.0/9

# mongo-express web UI (8081) - public and read-only (see above), so anyone
# with the link can browse results without GCP access
gcloud compute firewall-rules create allow-mongo-express \
  --network=default --direction=INGRESS --action=ALLOW \
  --rules=tcp:8081 --source-ranges=0.0.0.0/0

# get the internal IP Dataproc will connect to, and the external IP for mongo-express
export MONGO_INTERNAL_IP="$(gcloud compute instances describe "$MONGO_VM" --zone="$ZONE" --format='get(networkInterfaces[0].networkIP)')"
export MONGO_EXTERNAL_IP="$(gcloud compute instances describe "$MONGO_VM" --zone="$ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo "mongo-express (public, read-only): http://$MONGO_EXTERNAL_IP:8081"
```

> **Note:** `MONGO_EXTERNAL_IP` changes if the VM is ever deleted and
> recreated - update the link in your README when that happens (it stays
> fixed across simple reboots).

---

## 3. Dataproc - initialization action for pymongo/scikit-learn

Both Dataproc cluster commands below (interactive cluster in Part 5, managed
cluster in the Workflow Template in Part 4) install `pymongo` and
`scikit-learn` on every node via the tiny self-hosted init script uploaded in
Part 1 (`gs://$BUCKET/code/init-pip-install.sh`), and get `DATA_ROOT` /
`MONGO_URI` wired in as cluster-wide env vars via the `spark-env:` properties
prefix - this is what lets `src/pipeline/paths.py` and
`src/pipeline/mongo_export.py` switch from local to cloud with zero code
changes.

```bash
export PIP_INIT_ACTION="gs://$BUCKET/code/init-pip-install.sh"
export CLUSTER_PROPERTIES="spark-env:DATA_ROOT=gs://$BUCKET/data,spark-env:MONGO_URI=mongodb://writer:$MONGO_WRITER_PASSWORD@$MONGO_INTERNAL_IP:27017/market_analytics?authSource=market_analytics"
```

---

## 4. Dataproc Workflow Template - the one-click pipeline run

This is the artifact that makes the whole pipeline re-runnable from the
Console with a single "Run" click (Dataproc → Workflow Templates →
`market-analytics-pipeline` → Run): it provisions its own cluster, runs all
7 jobs in dependency order, and deletes the cluster when done - so nothing
sits around costing money between runs.

```bash
gcloud dataproc workflow-templates create market-analytics-pipeline --region="$REGION"

gcloud dataproc workflow-templates set-managed-cluster market-analytics-pipeline \
  --region="$REGION" \
  --cluster-name=market-analytics-wf \
  --master-machine-type=e2-standard-4 \
  --worker-machine-type=e2-standard-4 \
  --num-workers=2 \
  --master-boot-disk-size=50GB \
  --worker-boot-disk-size=50GB \
  --image-version=2.1-debian11 \
  --initialization-actions="$PIP_INIT_ACTION" \
  --properties="$CLUSTER_PROPERTIES"

# --- pipeline DAG: ingest -> features -> split -> {regression, clustering, association_rules, direction_labels} -> classification ---

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/ingest.py" \
  --step-id=ingest --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip"

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/microstructure.py" \
  --step-id=features --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip" --start-after=ingest

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/split.py" \
  --step-id=split --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip" --start-after=features

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/direction_labels.py" \
  --step-id=direction_labels --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip" --start-after=split

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/regression.py" \
  --step-id=regression --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip" --start-after=split

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/clustering.py" \
  --step-id=clustering --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip" --start-after=split

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/association_rules.py" \
  --step-id=association_rules --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip" --start-after=split

gcloud dataproc workflow-templates add-job pyspark "gs://$BUCKET/code/classification.py" \
  --step-id=classification --workflow-template=market-analytics-pipeline --region="$REGION" \
  --py-files="gs://$BUCKET/code/code.zip" --start-after=direction_labels
```

Run it (also doable from the Console - Dataproc → Workflow Templates → Run):

```bash
gcloud dataproc workflow-templates instantiate market-analytics-pipeline --region="$REGION"
```

---

## 5. Interactive Dataproc cluster - Jupyter via Component Gateway

A second, separate cluster purely for browsing results interactively
afterwards (the workflow template's cluster above is ephemeral and deletes
itself). Create it when needed, delete it when not - it's billed by the hour.

> **Trial-account disk quota**: a project on a free-trial/course quota may
> only have ~1000GB of persistent-disk quota in the region, while Dataproc's
> default boot disk is 1000GB *per node* (3000GB for 1 master + 2 workers).
> Spark reads/writes data via GCS, not local disk, so the boot disk only
> needs to hold the OS/Spark install + shuffle spill - `--master-boot-disk-size`
> / `--worker-boot-disk-size` below shrink it to 50GB/node (150GB total) to
> stay well under quota. If a cluster creation half-fails on quota, delete it
> before retrying (`gcloud dataproc clusters delete "$CLUSTER_JUPYTER"
> --region="$REGION" --quiet`) - a failed attempt can still leave the cluster
> name registered, which then errors with `ALREADY_EXISTS` on retry.

```bash
gcloud dataproc clusters create "$CLUSTER_JUPYTER" \
  --region="$REGION" --zone="$ZONE" \
  --master-machine-type=e2-standard-4 \
  --worker-machine-type=e2-standard-4 \
  --num-workers=2 \
  --master-boot-disk-size=50GB \
  --worker-boot-disk-size=50GB \
  --image-version=2.1-debian11 \
  --optional-components=JUPYTER \
  --enable-component-gateway \
  --initialization-actions="$PIP_INIT_ACTION" \
  --metadata='PIP_PACKAGES=pymongo scikit-learn' \
  --properties="$CLUSTER_PROPERTIES"

# get the clickable JupyterLab link (also visible on the cluster's Console page,
# under "Web Interfaces")
gcloud dataproc clusters describe "$CLUSTER_JUPYTER" --region="$REGION" \
  --format='value(config.endpointConfig.httpPorts."JupyterLab")'
```

Open that link, then use JupyterLab's GCS browser panel to open notebooks
directly from `gs://$BUCKET/notebooks/` - no manual download/upload needed.

Delete it after use:

```bash
gcloud dataproc clusters delete "$CLUSTER_JUPYTER" --region="$REGION" --quiet
```

---

## 6. Cleanup (tear down everything, stop billing)

```bash
gcloud dataproc clusters delete "$CLUSTER_JUPYTER" --region="$REGION" --quiet 2>/dev/null
gcloud compute instances delete "$MONGO_VM" --zone="$ZONE" --quiet
gcloud compute firewall-rules delete allow-mongo-internal allow-mongo-express --quiet
gsutil -m rm -r "gs://$BUCKET"
```

(The workflow template's managed cluster deletes itself automatically after
each run - nothing to clean up there. The template definition itself costs
nothing to leave in place; delete it with
`gcloud dataproc workflow-templates delete market-analytics-pipeline --region="$REGION" --quiet`
if you want it gone too.)

---

## Access for your teacher

The simplest path needs **zero GCP access** - see the README's "For my
teacher / reviewers" section: the results notebooks are committed to the
repo with their outputs baked in (GitHub renders `.ipynb` natively), and
`http://$MONGO_EXTERNAL_IP:8081` (mongo-express, Part 2) is public and
read-only, so anyone with the link can browse the curated results documents
each task wrote. Send the repo link and the mongo-express link; that's it.

If your teacher instead wants to **re-run the pipeline themselves** or use
the interactive JupyterLab cluster (Part 5) rather than the notebooks as
committed, they'll need access to your GCP project, granted from
**IAM & Admin → IAM → Grant Access** in the Console:

- `roles/dataproc.editor` - lets them open Workflow Templates and click Run,
  and open the Jupyter cluster's Component Gateway link.
- `roles/storage.objectViewer` - lets them browse `gs://$BUCKET` (code, data,
  notebooks) in the Console.

With that, from the Console they can open **Dataproc → Workflow Templates →
market-analytics-pipeline → Run** to execute the whole pipeline from
scratch, or open the `$CLUSTER_JUPYTER` cluster's JupyterLab link (Part 5) -
note that cluster isn't left running by default (it's billed hourly), so
you'd need to create it first (Part 5) before sharing that link.
