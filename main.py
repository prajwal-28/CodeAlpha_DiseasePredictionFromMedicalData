from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import sys

if __name__=="__main__":
    try:
        # 1. Ingestion
        print(">> Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print(f"Ingestion Completed. Data at: {train_data_path}")

        # 2. Transformation
        print(">> Starting Data Transformation...")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        print("Transformation Completed.")

        # 3. Training
        print(">> Starting Model Training...")
        model_trainer = ModelTrainer()
        accuracy, best_model_name = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Training Completed. Best Model: {best_model_name} with Accuracy: {accuracy*100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Ideally use logging here as well
