import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)



# Load artifacts
scaler = joblib.load('model/scaler.pkl')
TEST_DATA_PATH = "artifacts/test_data.csv"
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
st.subheader("Sample Test Dataset")

st.write(
    "You can download a sample test dataset (with ground truth labels) "
    "to understand the expected CSV format."
)

try:
    with open(TEST_DATA_PATH, "rb") as f:
        st.download_button(
            label="Download test_data.csv",
            data=f,
            file_name="test_data.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning("Sample test dataset not found in the repository.")

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


        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader("Evaluation Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")
        col4.metric("F1-Score", f"{f1:.3f}")
        col5.metric("ROC AUC", f"{roc_auc:.3f}")
        col6.metric("MCC", f"{mcc:.3f}")

        # Metrics
        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())