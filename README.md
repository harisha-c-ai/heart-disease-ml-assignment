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

## Results and Observations : TODO

