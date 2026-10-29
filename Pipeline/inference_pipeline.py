from zenml import pipeline

from Steps.ingest_data import ingest_data
from Steps.clean_data import clean_data
from Steps.predictor import predictor


@pipeline(enable_cache=False)
def inference_pipeline(data_path: str, model_path: str = "Saved_model/model"):
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    predictor(x_test, model_path)
