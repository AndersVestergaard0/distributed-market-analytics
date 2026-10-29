# Distributed Market Analytics

Cloud-native PySpark pipeline for high-frequency volatility forecasting on the
[Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
dataset. Built for INFS3208 (Cloud Computing) at UQ.

## Results - no setup required

- **Live results database (mongo-express)** - currently taken down to avoid
  ongoing GCE VM costs. The notebooks below cover the same results with
  their outputs already baked in; redeploy via
  [GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md) Part 2 to bring it back.
- **Results notebooks** (rendered on GitHub, no download needed):
  [Regression](notebooks/Regression_Results.ipynb) ·
  [Clustering](notebooks/Clustering_Results.ipynb) ·
  [Classification](notebooks/Classification_Results.ipynb) ·
  [Association rules](notebooks/AssociationRules_Results.ipynb) ·
  [EDA](notebooks/EDA.ipynb)

## Running locally

**1. Prerequisites**

- Python 3 + `pip install -r requirements.txt`
- **Java 17** - PySpark 3.5.1 does not support Java 21+ (newer JDKs break
  Hadoop's `UserGroupInformation` init with `getSubject is not supported`).
  If your default JDK is newer:
  ```bash
  brew install --cask temurin@17
  export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home
  ```
  Add the `export` line to your shell profile (`~/.bash_profile` or
  `~/.zshrc`) so it persists across terminal sessions.

**2. Get the data**

`data/raw/` and `data/processed/` are gitignored (not in the repo - too
large). Download the Kaggle dataset into `data/raw/`:

```bash
pip install kaggle
kaggle datasets download akshaymairal/optiver-realized-volatility-prediction -p data/raw --unzip
```

**3. Run the pipeline, in order**

Each step reads Parquet written by the one before it - run from `src/pipeline`
or `src/models`/`src/features` since the scripts resolve imports relative to
their own directory:

```bash
python src/pipeline/ingest.py          # raw book/trade/labels -> cleaned Parquet
python src/features/microstructure.py  # cleaned Parquet -> per-window engineered features
python src/models/split.py             # chronological train/val/test split + CV folds
python src/features/direction_labels.py  # first-half features + direction label (for classification)

python src/models/regression.py        # Task 1
python src/models/classification.py    # Task 2
python src/models/clustering.py        # Task 3
python src/models/association_rules.py # Task 4
```

Results print to stdout and are written to `data/models/`. Writing to the
results MongoDB is optional - without a `MONGO_URI` env var set, each script
just skips that step and logs `MONGO_URI not set - skipping write`.

## Cloud deployment

Deployed on GCP: GCS (data lake) → Dataproc (Spark, run as a one-click
Workflow Template) → MongoDB on a GCE VM (results store) → Jupyter via
Dataproc Component Gateway (visualization). Full setup commands in
[GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md).

![Architecture diagram](docs/architecture.png)