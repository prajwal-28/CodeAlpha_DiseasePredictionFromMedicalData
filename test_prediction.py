from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Creating a patient using REAL values from the CSV file:
patient = CustomData(
    age=63,
    sex='Male',              # Passed as String
    cp='typical angina',     # Passed as String
    trestbps=145,
    chol=233,
    fbs=True,                # Passed as Boolean
    restecg='lv hypertrophy',# Passed as String
    thalch=150,
    exang=False,             # Passed as Boolean
    oldpeak=2.3,
    slope='downsloping',     # Passed as String
    ca=0.0,
    thal='fixed defect'      # Passed as String
)

patient_df = patient.get_data_as_data_frame()
print("Patient Data:")
print(patient_df)

print("\nPredicting...")
predictor = PredictPipeline()
result = predictor.predict(patient_df)

print(f"Prediction: {result}")
print(f"Diagnosis: {'HEART DISEASE' if result[0] == 1 else 'HEALTHY'}")
