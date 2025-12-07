import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training (self, train_array, test_array):
        try:
            logging.info("Splitting Dependent And Independent Data From Train And Test Data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )

            ## Train Multiple Models

            models={
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print (model_report)
            print ("="*35)
            logging.info(f'Model Report: {model_report}')

            best_model_result = max(list(model_report.values()))
            # best_model = max(model_report, key=model_report.key)
            best_model_name = [k for k, v in model_report.items() if v==best_model_result][0]

            best_model = models[best_model_name]
            
            print (f"Best Model Found, Model Name: {best_model_name}, R2 _Score: {best_model_result}")
            print ("="*35)
            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 _Score: {best_model_result}")
            logging.info("Hyperparameter Tuning Started For CatBoost.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Exception Occured In Model Training")
            raise CustomException(e, sys)