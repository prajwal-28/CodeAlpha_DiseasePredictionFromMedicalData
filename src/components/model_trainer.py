import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": GradientBoostingClassifier(), # Using GradientBoosting as a placeholder or addition
                "Logistic Regression": LogisticRegression(),
                "XGBoost": XGBClassifier(),
                "SVC": SVC(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }
            
            # Dictionary filtering to match strict request if needed, but imports suggest using all.
            # Only keeping the ones explicitly requested + imports
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBoost": XGBClassifier(),
                "SVC": SVC(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            model_report: dict = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                test_model_score = accuracy_score(y_test, y_test_pred)

                model_report[list(models.keys())[i]] = test_model_score

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            
            logging.info(f"Best Model: {best_model_name} with Accuracy: {acc_score}")

            return acc_score, best_model_name

        except Exception as e:
            raise CustomException(e, sys)
