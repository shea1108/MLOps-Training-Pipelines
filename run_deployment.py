import os

import click

from Pipeline.deployment_pipeline import continuous_deployment_pipeline
from Pipeline.inference_pipeline import inference_pipeline

DATA_PATH = "data/olist_customers_dataset.csv"
MODEL_PATH = "Saved_model/model"

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
)
@click.option("--min-r2", default=0.0, help="R2_Score threshold minium is: ")
def main(config: str, min_r2: float):
    if config in (DEPLOY, DEPLOY_AND_PREDICT):
        continuous_deployment_pipeline(
            data_path=DATA_PATH, min_r2=min_r2, model_path=MODEL_PATH
        )

    if config in (PREDICT, DEPLOY_AND_PREDICT):
        if not os.path.exists(MODEL_PATH):
            print(f"No model has been deployed at {MODEL_PATH}. Try to running --config deploy.")
            return
        inference_pipeline(data_path=DATA_PATH, model_path=MODEL_PATH)


if __name__ == "__main__":
    main()
