import streamlit as st
import pandas as pd
import joblib




model = joblib.load('logistic_heart.pkl')
scaler = joblib.load('scaler_heart.pkl')
expected_columns = joblib.load('columns_heart.pkl') 

st.title("Heart Disease Prediction by Roshan Kumar ❤️")
st.markdown("This app predicts the likelihood of a heart disease based on user input features.")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    raw_data = {
        "Age": age,
        "Sex_" + sex: 1,
        "ChestPainType _" + chest_pain : 1,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG_" + resting_ecg: 1,
        "MaxHeartRate": max_heart_rate,
        "ExerciseAngina_" + str(exercise_angina): 1,
        "OldPeak": oldpeak,
        "ST_Slope_" + st_slope: 1
    }
    input_df= pd.DataFrame([raw_data])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    input_df_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df_scaled)[0]

    if prediction == 1:
        st.error("High risk of heart attack/Disease. Please consult a healthcare professional.")
    else:
        st.success("Low risk of heart attack/Disease.  ")

