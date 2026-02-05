from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create dummy patient data (Using EXACT column names like 'thalch')
patient = CustomData(
    age=60, 
    sex=1, 
    cp=0, 
    trestbps=130, 
    chol=250, 
    fbs=0, 
    restecg=1, 
    thalch=150, 
    exang=0, 
    oldpeak=2.5, 
    slope=0, 
    ca=0, 
    thal=1
)

patient_df = patient.get_data_as_data_frame()
print("Patient Data for Prediction:")
print(patient_df)

print("Predicting...")
predictor = PredictPipeline()
result = predictor.predict(patient_df)

print(f"Raw Prediction: {result}")
print(f"Diagnosis: {'HEART DISEASE DETECTED' if result[0] == 1 else 'NORMAL'}")
