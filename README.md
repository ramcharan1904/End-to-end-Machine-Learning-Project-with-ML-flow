# Wine Quality Prediction — End-to-End ML Pipeline

A reproducible machine learning pipeline that predicts wine quality from physicochemical
properties using an ElasticNet regression model, with full experiment tracking, containerized
deployment, and CI/CD to AWS.

## What it does

Predicts a wine's quality score (0–10) from 11 physicochemical properties: fixed acidity,
volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur
dioxide, density, pH, sulphates, and alcohol (UCI Wine Quality dataset).

The model is an **ElasticNet regression** (`alpha=0.1`, `l1_ratio=0.2`), chosen for its combined
L1/L2 regularization, which handles the correlated, low-dimensional feature set in this dataset
well without overfitting.

**Results:** `[ADD: validation/test RMSE, MAE, and R² from your MLflow run]`

## Architecture

```
Raw data → Schema validation (schema.yaml) → Train/test split
        → ElasticNet training (params.yaml) → Evaluation, logged to MLflow (via DagsHub)
        → Flask inference app (app.py) → Docker container
        → Pushed to AWS ECR → Deployed on EC2 via GitHub Actions
```

## Tech stack

Python, scikit-learn, MLflow, DagsHub, Flask, Docker, AWS (EC2, ECR), GitHub Actions

## Project structure

```
config/             # config.yaml — paths and pipeline settings
params.yaml         # model hyperparameters
schema.yaml         # expected input schema and target column
src/mlproject/      # pipeline components: ingestion, validation, transformation, training, evaluation
research/           # exploratory notebooks
app.py              # Flask inference app
main.py             # pipeline entry point
Dockerfile          # container definition for deployment
```

## Running it locally

```bash
git clone https://github.com/ramcharan1904/Wine-Tasting-End-to-End
cd Wine-Tasting-End-to-End

conda create -n mlproj python=3.8 -y
conda activate mlproj

pip install -r requirements.txt

python main.py    # runs the training pipeline
python app.py     # starts the Flask app — open the local host/port shown in the terminal
```

## Experiment tracking (MLflow + DagsHub)

Runs are tracked remotely via [DagsHub](https://dagshub.com). **Never commit real credentials —
set these as environment variables instead:**

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/Wine-Tasting-End-to-End.mlflow
export MLFLOW_TRACKING_USERNAME=<your_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>

mlflow ui
```

## Deployment (AWS EC2 + GitHub Actions)

On push to `main`, GitHub Actions builds the Docker image, pushes it to Amazon ECR, and a
self-hosted runner on the EC2 instance pulls the latest image and restarts the container.

Required IAM policies: `AmazonEC2ContainerRegistryFullAccess`, `AmazonEC2FullAccess`

Required GitHub repo secrets (**Settings → Secrets → Actions** — never hardcode these anywhere
in the repo):

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_ECR_LOGIN_URI
ECR_REPOSITORY_NAME
```

## What I'd improve next

`[ADD: e.g., compare against a tree-based model (Random Forest / XGBoost) as a baseline,
add automated tests for the pipeline components, add a model registry stage in MLflow]`

## Dataset

[UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) —
physicochemical properties and expert quality ratings for red and white wine samples.
