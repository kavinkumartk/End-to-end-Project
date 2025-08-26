import streamlit as st
import joblib
import numpy as np

# Load model and scaler together
model, scaler = joblib.load("destree_model.sav")

st.title("📘 Student Exam Pass/Fail Predictor")
st.write("Enter the details below to check if the student passes the exam:")

hours_studied = st.number_input("📖 Hours Studied", min_value=0.0, max_value=12.0, value=0.0, step=0.5)
sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=0.0, step=0.5)
attendance = st.number_input("🏫 Attendance (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

features = np.array([[hours_studied, sleep_hours, attendance]])
features_scaled = scaler.transform(features)  # ✅ scale before predict

if st.button("🔮 Predict Result"):
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.success("✅ The student is likely to **PASS** 🎉")
    else:
        st.error("❌ The student is likely to **FAIL** 😢")
