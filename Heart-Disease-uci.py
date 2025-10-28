# UCI (name: khaled waled talat)

import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# =========================
# (Theme)
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction ❤️",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS 
st.markdown("""
    <style>
    .main {
        background-color: #f9fafc;
        color: #222;
    }
    h1, h2, h3 {
        text-align: center;
        color: #d90429;
    }
    .stButton>button {
        background-color: #d90429;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ef233c;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# modle genirative
# =========================
model_path = os.path.join(os.path.dirname(__file__), "final_model.pkl")

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    st.error("❌ Model file not found! Please make sure 'final_model.pkl' exists in the same folder as this script.")
    st.stop()

# =========================
# data adress
# =========================
st.title("❤️ Heart Disease Prediction")
st.markdown("### Provide the patient's details to predict the risk of heart disease.")

# =========================
# data entry
# =========================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    sex = st.selectbox("gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"])

with col2:
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# =========================
# data gineration
# =========================
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {"Yes": 1, "No": 0}
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

input_data = np.array([[ 
    age,
    sex_map[gender],
    cp_map[cp],
    trestbps,
    chol,
    fbs_map[fbs],
    restecg_map[restecg],
    thalach,
    exang_map[exang],
    oldpeak,
    slope_map[slope],
    ca,
    thal_map[thal]
]])

# =========================
# predict
# =========================
if st.button("🔍Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # Percentage

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease detected ({probability:.1f}% risk).")
    else:
        st.success(f"✅ No significant signs of heart disease detected ({probability:.1f}% risk).")

    # =========================
    # (Gauge Chart)
    # =========================
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        title={'text': "Heart Disease Risk (%)", 'font': {'size': 22}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if probability > 50 else "green"},
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 100], 'color': 'lightcoral'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': probability}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
