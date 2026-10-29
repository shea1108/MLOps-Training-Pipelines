from zenml import pipeline

from Steps.ingest_data import ingest_data
from Steps.clean_data import clean_data
from Steps.train_model import train_model
from Steps.evaluations import evaluation
from Steps.deployment_trigger import deployment_trigger
from Steps.model_deployer import model_deployer
from Steps.config import ModelNameConfig


@pipeline(enable_cache=True)
def continuous_deployment_pipeline(
    data_path: str,
    min_r2: float = 0.0,
    model_path: str = "Saved_model/model",
):
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test, ModelNameConfig())
    rmse, mae, r2_score = evaluation(model, x_test, y_test)
    deploy_decision = deployment_trigger(r2_score, min_r2)
    model_deployer(model, deploy_decision, model_path)
