import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Title
st.title("Heart Disease Prediction System")

# Form
with st.form("prediction_form"):
    st.write("Enter Patient Details:")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=60, step=1)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        cp = st.selectbox("Chest Pain Type (cp)", ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'])
        trestbps = st.number_input("Resting BP (trestbps)", min_value=50.0, max_value=250.0, value=130.0, step=1.0)
        chol = st.number_input("Cholesterol", min_value=100.0, max_value=600.0, value=250.0, step=1.0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [True, False])
        restecg = st.selectbox("Resting ECG (restecg)", ['lv hypertrophy', 'normal', 'st-t abnormality'])

    with col2:
        thalch = st.number_input("Max Heart Rate (thalch)", min_value=50.0, max_value=250.0, value=150.0, step=1.0)
        exang = st.selectbox("Exercise Induced Angina (exang)", [True, False])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
        slope = st.selectbox("Slope", ['downsloping', 'flat', 'upsloping'])
        ca = st.number_input("Number of Major Vessels (ca) (0-3)", min_value=0.0, max_value=3.0, value=0.0, step=1.0)
        thal = st.selectbox("Thal", ['fixed defect', 'normal', 'reversable defect'])

    submit_button = st.form_submit_button("Predict")

    if submit_button:
        try:
            # Instantiate CustomData with form values
            data = CustomData(
                age=int(age),
                sex=sex,
                cp=cp,
                trestbps=float(trestbps),
                chol=float(chol),
                fbs=fbs,
                restecg=restecg,
                thalch=float(thalch),
                exang=exang,
                oldpeak=float(oldpeak),
                slope=slope,
                ca=float(ca),
                thal=thal
            )
            
            # Get data as dataframe
            df = data.get_data_as_data_frame()
            
            # Instantiate PredictPipeline
            predictor = PredictPipeline()
            result = predictor.predict(df)
            
            # Display Result
            if result[0] == 1:
                st.error("HEART DISEASE DETECTED")
            else:
                st.success("HEALTHY")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
