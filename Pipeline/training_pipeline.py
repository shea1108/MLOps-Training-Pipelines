import logging
import pandas as pd
from Steps.ingest_data import ingest_data
from Steps.clean_data import clean_data
from Steps.evaluations import evaluation
from Steps.train_model import train_model
from Steps.config import ModelNameConfig


from zenml import pipeline


@pipeline(enable_cache=True)
def training_pipelines(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    rmse, mae, r2_score = evaluation(model, X_test, y_test)
    return rmse, mae, r2_score
    
