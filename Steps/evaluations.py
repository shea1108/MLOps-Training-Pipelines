import logging
import pandas as pd
import numpy as np
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from sklearn.base import RegressorMixin
from src.evaluation import MAE, RMSE, R2
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker= experiment_tracker.name)
def evaluation(model:RegressorMixin, x_test:pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float,"rmse"], Annotated[float, "mae"] ,Annotated[float, "r2_score"]] :

    try:    
        model_prediction = model.predict(x_test)
        rmse = RMSE().calculate_score(y_test, model_prediction)
        mlflow.log_metric("rmse", rmse)

        mae = MAE().calculate_score(y_test, model_prediction)
        mlflow.log_metric("mae", mae)

        r2_score = R2().calculate_score(y_test, model_prediction)
        mlflow.log_metric("r2_score", r2_score)

        return rmse, mae, r2_score
    except Exception as e:
        logging.error(e)
        raise e
