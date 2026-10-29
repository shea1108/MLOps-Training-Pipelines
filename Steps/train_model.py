import logging
import pandas as pd
import mlflow

from zenml.client import Client
from zenml import step
from sklearn.base import RegressorMixin
from zenml.client import Client
from Steps.config import ModelNameConfig
from src.model_dev import (
    HyperparameterTuner,
    LinearRegressionModel,
    RandomForestModel,
    XGboostModel
)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker= experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    print(f"Training model: {config.model_name}")
    if config.model_name == "LinearRegression":
        mlflow.sklearn.autolog()
        model = LinearRegressionModel()
        trained_model = model.train(x_train, y_train)
        mlflow.sklearn.autolog(disable=True)
        return trained_model
    else:
        raise ValueError("Model {} not supported".format(config.model_name)) 