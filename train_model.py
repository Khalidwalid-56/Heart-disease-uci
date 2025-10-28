
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay
import seaborn as sns



import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# luad the url

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

data = pd.read_csv(url, names=columns)
data = data.replace('?', np.nan)

# convert all columns to numeric
for c in columns:
    data[c] = pd.to_numeric(data[c], errors='coerce')

# convert target to binary (0 = no disease, 1 = disease)
data['target'] = (data['target'] > 0).astype(int)

print(" Data Overview ")
print(data.head())
print(data.info())
print(data.describe())



imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


#  Split Data

X = data_imputed.drop('target', axis=1)
y = data_imputed['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}
   
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
    plt.title(f"ROC Curve - {name}")
    plt.show()
    

print("\n UCI Project completed successfully ")





# حفظ أفضل موديل (Random Forest هنا)
best_model = models["Random Forest"]

with open("final_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Model saved as final_model.pkl!")
