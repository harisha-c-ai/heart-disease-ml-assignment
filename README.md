# Heart Disease Classification – ML Assignment 2

## Problem Statement
The objective of this project is to develop a predictive model that can accurately classify whether a patient has heart disease based on various health metrics and demographic information. By analyzing features such as age, cholesterol levels, and other clinical indicators, the model aims to assist healthcare professionals in early diagnosis and timely intervention for heart disease.

## Dataset Description
- **Dataset:** Heart Disease Dataset
- **Source:** https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Number of Samples:** 1025
- **Number of Features:** 13
- **Target Variable:** `target`
  - 0 → No heart disease
  - 1 → Heart disease present

## Models Implemented
The following machine learning models were implemented and evaluated:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes - GaussianNB
- Random Forest
- XGBoost

To ensure code reusability and consistency, a common `BaseModel` class was introduced.  
This base class encapsulates shared functionality such as model training, prediction, and evaluation using standard classification metrics. Each individual model inherits from this base class and defines only model-specific initialization logic.

This design avoids code duplication, improves maintainability, and ensures fair and consistent evaluation across all models.

## Evaluation Metrics
The models were evaluated using the following metrics:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

## Project Structure

```bash
heart-disease-ml-assignment/
│
├── app.py                           # Streamlit application for model evaluation
├── train.py                         # Script for training models
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
│
├── models/                          # Model implementations & saved models
│   ├── base_model.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes_gaussian.py
│   ├── random_forest.py
│   ├── xgboost.py
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── gaussian_naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
└── artifacts/                       # Training outputs & evaluation artifacts
    ├── model_results.csv
    └── test_data.csv
```

  

## Results and Observations

The performance of all implemented models was evaluated on a held-out test dataset using standard classification metrics. The results are summarized in the table below.

| Model Model Name    | Accuracy | AUC   | Precision | Recall | F1-Score | MCC   |
|---------------------|----------|-------|-----------|--------|----------|-------|
| Logistic Regression | 0.8098   | 0.9298| 0.7619    | 0.9143 | 0.8312   | 0.6309|
| Decision Tree       | 0.9854   | 0.9857| 1.0000    | 0.9714 | 0.9855   | 0.9712|
| KNN                 | 0.8634   | 0.9629| 0.8738    | 0.8571 | 0.8654   | 0.7269|
| Naive Bayes         | 0.8293   | 0.9043| 0.8070    | 0.8762 | 0.8402   | 0.6602|
| Random Forest       | 1.0000   | 1.0000| 1.0000    | 1.0000 | 1.0000   | 1.0000|
| XGBoost             | 0.9610   | 0.9900| 0.9533    | 0.9714 | 0.9623   | 0.9220|

## Model-wise Observations

| **ML Model Name** | **Observation about model performance** |
|------------------|-------------------------------------------|
| **Logistic Regression** | Logistic Regression provides a strong baseline model with high recall, indicating good ability to identify patients with heart disease. However, its lower accuracy and MCC suggest limited capability in capturing complex non-linear relationships in the data. |
| **Decision Tree** | Decision Tree achieves very high accuracy and MCC, indicating excellent performance on the test set. However, such near-perfect results suggest a risk of overfitting, as decision trees can easily memorize training patterns. |
| **kNN** | KNN shows balanced performance with good Accuracy, AUC, and MCC. While it generalizes better than a single Decision Tree, its Recall is slightly lower than Logistic Regression, making it less optimal when minimizing false negatives is critical. |
| **Naive Bayes** | Naive Bayes performs reasonably well with good recall but lower AUC and MCC compared to other models. Its performance is limited by the assumption of feature independence, which may not hold for medical datasets. |
| **Random Forest (Ensemble)** | Random Forest achieves the best performance across all evaluation metrics, demonstrating the strength of ensemble learning in capturing complex patterns. However, perfect scores should be interpreted cautiously as they may indicate optimistic evaluation or overfitting. |
| **XGBoost (Ensemble)** | XGBoost shows excellent performance with very high accuracy, AUC, and MCC. It generalizes better than a single decision tree and effectively handles non-linear relationships, making it a strong and reliable ensemble model. |


## Streamlit Application

This project includes a Streamlit web application that allows users to evaluate trained machine learning models on test data.

### How to Run the App Locally

1. Clone the repository:
```bash
git clone https://github.com/harisha-c-ai/heart-disease-ml-assignment.git
cd heart-disease-ml-assignment
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
./venv/Scripts/activate  # Windows
source venv/bin/activate  # macOS/Linux
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

created with ❤️ by Harisha.