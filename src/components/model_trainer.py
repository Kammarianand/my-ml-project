import os
import sys

from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_traininer(self, train_array, test_array):
        try:
            logging.info('splitting training and testing input data')
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                'RandomForest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighbors': KNeighborsRegressor(),
                'SVR': SVR(),
                'DecisionTree': DecisionTreeRegressor(),
                'XGB': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False)
            }

            params = {
                "DecisionTree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "RandomForest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "GradientBoosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "LinearRegression": {},
                "KNeighbors": {
                    'n_neighbors': [3, 5, 7, 9]
                },
                "SVR": {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto']
                },
                "XGB": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')

            logging.info("best model found on both training and testing data")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e, sys)