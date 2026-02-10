import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report


# Load artifacts
scaler = joblib.load('model/scaler.pkl')
MODELS = {
    "Logistic Regression": ("logistic_regression_model.pkl", True),
    "KNN": ("knn_model.pkl", True),
    "Decision Tree": ("decision_tree_model.pkl", False),
    "Naive Bayes": ("gaussian_naive_bayes_model.pkl", False),
    "Random Forest": ("random_forest_model.pkl", False),
    "XGBoost": ("xgboost_model.pkl", False),
}

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

st.set_page_config(page_title="Heart Disease Classification", layout="centered")
st.title("❤️ Heart Disease Classification App")
st.write("Upload a **test CSV file with ground truth labels (`target`)** to evaluate models.")

# Model selection
model_name = st.selectbox("Select Model", list(MODELS.keys()))
model_file, needs_scaling = MODELS[model_name]
model = joblib.load(f"model/{model_file}")

# CSV Upload
uploaded_file = st.file_uploader("Upload Test CSV (must include `target`)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X = df[FEATURE_COLUMNS]
        y_true = df["target"]

        if needs_scaling:
            X = scaler.transform(X)

        y_pred = model.predict(X)

        