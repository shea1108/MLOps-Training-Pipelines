# MLOps-Training-Pipelines

A hands-on MLOps project built while learning ZenML + MLflow: predicting Olist
e-commerce review scores with a full train -> evaluate -> deploy -> infer
pipeline.

## Project structure

```
Steps/        individual ZenML steps (ingest, clean, train, evaluate, deploy, predict)
Pipeline/     ZenML pipelines that wire the steps together
src/          plain Python model/data logic used by the steps
run_pipelines.py     entry point: run the training pipeline
run_deployment.py    entry point: run the deployment / inference pipelines
Datadown.py          helper to download the raw dataset via kagglehub
```

## Pipelines

- **`training_pipelines`** (`run_pipelines.py`): ingest -> clean -> train -> evaluate.
- **`continuous_deployment_pipeline`** (`run_deployment.py --config deploy`):
  same as above, then gates deployment on `r2_score > min_r2` and saves the
  model to `Saved_model/model` if it passes.
- **`inference_pipeline`** (`run_deployment.py --config predict`): loads the
  deployed model from `Saved_model/model` and runs predictions, without
  retraining.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The raw dataset is not committed to this repo. Place
`olist_customers_dataset.csv` at `data/olist_customers_dataset.csv` before
running any pipeline. You can fetch the full dataset with:

```bash
python Datadown.py
```

This downloads the "Brazilian E-Commerce" dataset via `kagglehub`; copy
`olist_customers_dataset.csv` from the downloaded path into `data/`.

## Running

```bash
python run_pipelines.py                              # train + evaluate
python run_deployment.py --config deploy              # train + evaluate + deploy if good enough
python run_deployment.py --config predict              # predict using the deployed model
python run_deployment.py --config deploy_and_predict   # both, in order
```

## MLflow UI

Experiment tracking is handled by a ZenML MLflow experiment tracker. To
inspect runs locally:

```bash
.venv\Scripts\mlflow.exe ui --backend-store-uri <tracking-uri-from-your-stack> --port 5000
```
