import logging 
import pandas as pd
import xgboost as xgb
import optuna

from abc import ABC, abstractmethod
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod    
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        pass


class RandomForestModel(Model):

    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg     

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1 , 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        return reg.score(x_test,y_test)
    

class XGboostModel(Model):

    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train,y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_loginiform("learning_rate", 1e-7, 10.0)
        reg = self.train(x_train, y_train, n_estimators= n_estimators, max_depth= max_depth, learning_rate=learning_rate)
        return reg



class LinearRegressionModel(Model):

    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train,y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.optimize(x_train, y_train)
        return reg.score (x_test,y_test)


class HyperparameterTuner:

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def optimize(self, n_trials= 100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_params.params
