import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    layout="wide",
    page_icon="❤️"
)
st.markdown("""
<style>     
body {
    background-color:#0f172a;
    color:white;
}

.main-title {
    text-align:center;
    font-size:28px;
    font-weight:600;
    margin-bottom:5px;
    color:#fffff;
}

.metric-card {
    padding:5px;
    border-radius:10px;
    color:white;
    text-align:center;
    font-size:10px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.4);
}

.metric-value {
    font-size:20px;
    font-weight:bold;
}

.green {background:linear-gradient(45deg,#1b5e20,#43a047);}
.blue {background:linear-gradient(45deg,#0d47a1,#42a5f5);}
.orange {background:linear-gradient(45deg,#e65100,#fb8c00);}
.purple {background:linear-gradient(45deg,#4a148c,#8e24aa);}
.teal {background:linear-gradient(45deg,#004d40,#26a69a);}
.darkblue {background:linear-gradient(45deg,#1a237e,#3949ab);}

hr {
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# load scaler and models
scaler = joblib.load("model/scaler.pkl")
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

st.sidebar.title("⚙️ Controls")

model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
model_file, needs_scaling = MODELS[model_name]
model = joblib.load(f"model/{model_file}")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must include target)", type=["csv"]
)

# Download sample test data
try:
    with open(TEST_DATA_PATH, "rb") as f:
        st.sidebar.download_button(
            label="⬇ Download Sample Test Data",
            data=f,
            file_name="test_data.csv",
            mime="text/csv"
        )
except:
    st.sidebar.warning("Sample dataset not found.")

st.markdown(f"<div class='main-title'>Heart Disease Prediction Dashboard</div>", unsafe_allow_html=True)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain 'target' column.")
        st.stop()

    X = df[FEATURE_COLUMNS]
    y_true = df["target"]

    if needs_scaling:
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    # metric cards
    cols = st.columns(6)

    metrics_data = [
        ("Accuracy", f"{accuracy:.2%}", "green"),
        ("Precision", f"{precision:.2f}", "blue"),
        ("Recall", f"{recall:.2f}", "orange"),
        ("F1 Score", f"{f1:.2f}", "purple"),
        ("ROC AUC", f"{roc_auc:.2f}", "teal"),
        ("MCC", f"{mcc:.2f}", "darkblue")
    ]

    for col, (label, value, color) in zip(cols, metrics_data):
        col.markdown(f"""
        <div class="metric-card {color}">
            {label}
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.space(1)

    left, right = st.columns(2)

    with left:
        st.markdown("##### Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease","Disease"],
            yticklabels=["No Disease","Disease"],
            cbar=False,
            ax=ax
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)

    
    with right:
        st.markdown("##### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, height="auto")
    

else:
    st.info("👈 Upload a test dataset from the sidebar to view dashboard results.")
    col1, col2, col3 = st.columns(3)

