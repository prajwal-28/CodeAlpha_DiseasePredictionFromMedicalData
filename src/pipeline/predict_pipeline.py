import sys
import os
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("models", "model.pkl")
            preprocessor_path = os.path.join("models", "preprocessor.pkl")
            print("Before Loading")
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 age: int,
                 sex: int,
                 cp: int,
                 trestbps: int,
                 chol: int,
                 fbs: int,
                 restecg: int,
                 thalch: int,
                 exang: int,
                 oldpeak: float,
                 slope: int,
                 ca: int,
                 thal: int):
        
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalch = thalch
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "cp": [self.cp],
                "trestbps": [self.trestbps],
                "chol": [self.chol],
                "fbs": [self.fbs],
                "restecg": [self.restecg],
                "thalch": [self.thalch],
                "exang": [self.exang],
                "oldpeak": [self.oldpeak],
                "slope": [self.slope],
                "ca": [self.ca],
                "thal": [self.thal],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
