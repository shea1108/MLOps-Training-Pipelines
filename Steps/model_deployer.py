import logging
import os
import shutil

import mlflow
from sklearn.base import RegressorMixin
from zenml import step


@step(enable_cache=False)
def model_deployer(
    model: RegressorMixin,
    deploy_decision: bool,
    model_path: str = "Saved_model/model",
) -> bool:
    if not deploy_decision:
        logging.info(
            f"r2_score not higher than threshold, skip deploy model."
        )
        return False

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    mlflow.sklearn.save_model(model, model_path)
    logging.info(f"Deployed model at: {model_path}")
    return True
