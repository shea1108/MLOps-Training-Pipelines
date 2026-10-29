import logging
import pandas as pd

from typing_extensions import Annotated
from typing import Tuple
from zenml import step
from src.data_cleaning import DataDivineStrategy, DataPreProcessStrategy, DataCleaning

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"],
]:
    try:
        """Data PreProcess"""
        strategy_preprocess = DataPreProcessStrategy()
        data_clean = DataCleaning(df, strategy_preprocess)
        data_cleanned = data_clean.handle_data()

        """Data Divine"""
        strategy_divine = DataDivineStrategy()
        data_divine = DataCleaning(data_cleanned, strategy_divine)
        X_train, X_test, Y_train, Y_test = data_divine.handle_data()
        logging.info("Data Divine Completed")
        return X_train, X_test, Y_train, Y_test
    except Exception as e:
        logging.error("Formatting error: {}".format(e))
        raise e





