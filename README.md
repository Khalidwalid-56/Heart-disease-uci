# Heart Disease Prediction ❤️

A machine learning project to predict the risk of heart disease using patient data. This project includes a Streamlit app, training scripts, and a trained model.

---

## Features

- **Data Source:** UCI Heart Disease dataset.
- **Models Trained:** Logistic Regression & Random Forest.
- **Prediction:** Predicts heart disease risk as binary (0 = No disease, 1 = Disease) and shows probability.
- **Visualization:** ROC curves and probability gauge chart using Plotly.
- **Streamlit App:** Interactive UI for entering patient details and predicting risk.

---

## Files

- `train_model.py` — Script to train the models and save the best model (`final_model.pkl`).
- `final_model.pkl` — Saved Random Forest model used in the app.
- `Heart-Disease-uci.py` — Streamlit app for heart disease prediction.
- `requirements.txt` — Python dependencies.

---

## How to Run

1. **Install dependencies**
```bash
pip install -r requirements.txt


Run the training script (optional)

python train_model.py


Run the Streamlit app

streamlit run Heart-Disease-uci.py
