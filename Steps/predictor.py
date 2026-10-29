import logging

import mlflow
import numpy as np
import pandas as pd
from zenml import step


@step(enable_cache=False)
def predictor(x_test: pd.DataFrame, model_path: str = "Saved_model/model") -> np.ndarray:
    model = mlflow.sklearn.load_model(model_path)
    predictions = model.predict(x_test)
    logging.info(f"Du doan {len(predictions)} dong, 5 dong dau: {predictions[:5]}")
    return predictions
