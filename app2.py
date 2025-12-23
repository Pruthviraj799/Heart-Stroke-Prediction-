import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System by Pruthviraj")
st.markdown(
    """
    This application predicts the **risk of heart disease** based on medical parameters.
    Please enter accurate details for better prediction.
    """
)

st.divider()

st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic"
    )
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.subheader("🧾 Prediction Result")

    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease**\n\nPlease consult a cardiologist for further evaluation.")
    else:
        st.success("✅ **Low Risk of Heart Disease**\n\nMaintain a healthy lifestyle and regular checkups.")

st.divider()

# Footer
st.markdown(
    """
    <div style="text-align:center; font-size:14px;">
    Developed by <b>Pruthviraj</b> | AI & ML Project
    </div>
    """,
    unsafe_allow_html=True
)
