import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the RMSE class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("The RMSE score is:" + str(rmse))
            return rmse
        except Exception as e:
            logging.error("Exception occurered in calculate_score method RMSE class" + str(e))
            return


class MAE(Evaluation):
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the MAE class")
            mae = mean_absolute_error(y_true,y_pred)
            logging.info("The MAE score is: "+str(mae))
            return mae

        except Exception as e:
            logging.error("The exception occurred in calculate_score method of MAE class:" + str(e))
            return e

class R2(Evaluation):
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the R2 score class")
            r2score = r2_score(y_true,y_pred)
            logging.info("The R2 score is: " + str(r2score))
            return r2score
            
        except Exception as e:
            logging.info("The exception occured in calculate_score method of R2:" + str(e))
            return e
