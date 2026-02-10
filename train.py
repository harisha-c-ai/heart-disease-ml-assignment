import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# model imports
from model.logistic_regression import LogisticRegressionModel
from model.decision_tree import DecisionTreeModel
from model.k_nearest_neighbors import KNNModel
from model.naive_bayes_gausian import NaiveBayesGaussianModel
from model.random_forest import RandomForestModel
from model.xgboost import XGBoostModel

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH = r'E:\MTech-Workspace\Playground\Assignments\ML\assignment2\heart.csv'
ARTIFACTS_DIR = "artifacts"
MODEL_DIR = "model"

results_df = pd.DataFrame(columns=[
    'Model',
    'Training Time (s)',
    'Accuracy',
    'AUC',
    'Precision',
    'Recall',
    'F1-Score',
    'MCC'
])
def store_results(model_name, training_time,
                  accuracy, auc, precision, recall, f1_score, mcc):
    """
    Appends model evaluation results to results_df
    """
    global results_df
    new_row = {
        'Model': model_name,
        'Training Time (s)': round(training_time, 4),
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1_score, 4),
        'MCC': round(mcc, 4)
    }
    
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # load dataset
    data = pd.read_csv(DATA_PATH)

    # Dataset information
    dataset_name = "Heart Disease Dataset"
    dataset_source = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
    n_samples = data.shape[0]      
    n_features = data.shape[1] - 1  
    problem_type = "binary_classification"

    print(f"Dataset: {dataset_name}")
    print(f"Source: {dataset_source}")
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"Problem Type: {problem_type}")

    # Preprocess
    # Separate features (X) and target (y)
    X = data.drop('target', axis=1)
    y = data['target']

    # Handle missing values if any
    if X.isnull().sum().any():
        print("Missing values found!.")
    else:
        print("No missing values found.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # save test split for later use in inference
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv(os.path.join(ARTIFACTS_DIR, "test_data.csv"), index=False)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler (needed for inference)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    train_samples = X_train_scaled.shape[0]       
    test_samples = X_test_scaled.shape[0]          
    train_test_ratio = train_samples / (train_samples + test_samples)  

    print(f"Train samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Split ratio: {train_test_ratio:.1%}")

    # --------------------------------------------------------------
    # training logistic regression model
    # --------------------------------------------------------------
    print("Training LogisticRegression model...")
    lr_model = LogisticRegressionModel(max_iter=1000, random_state=42)
    lr_model.train(X_train_scaled, y_train)

    print(f"Logistic Regression training completed in {lr_model.training_time:.2f}s")

    # Evaluate model
    lr_metrics = lr_model.evaluate(X_test_scaled, y_test)

    # Store results
    store_results(
        model_name="Logistic Regression",
        training_time=lr_model.training_time,
        accuracy=lr_metrics['accuracy'],
        auc=lr_metrics['roc_auc'],
        precision=lr_metrics['precision'],
        recall=lr_metrics['recall'],
        f1_score=lr_metrics['f1_score'],
        mcc=lr_metrics['mcc']
    )

    # save model
    lr_model.save_model(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))

    # --------------------------------------------------------------
    # training decision tree model
    # --------------------------------------------------------------
    print("Training Decision Tree model...")
    dt_model = DecisionTreeModel(random_state=42)
    dt_model.train(X_train, y_train)
    print(f"Decision Tree training completed in {dt_model.training_time:.2f}s")

    # Evaluate model
    dt_metrics = dt_model.evaluate(X_test, y_test)
    
    # Store results
    store_results(
        model_name="Decision Tree",
        training_time=dt_model.training_time,
        accuracy=dt_metrics['accuracy'],
        auc=dt_metrics['roc_auc'],
        precision=dt_metrics['precision'],
        recall=dt_metrics['recall'],
        f1_score=dt_metrics['f1_score'],
        mcc=dt_metrics['mcc']
    )

    # save model
    dt_model.save_model(os.path.join(MODEL_DIR, "decision_tree_model.pkl"))

    # --------------------------------------------------------------
    # training KNN model
    # --------------------------------------------------------------
    print("Training KNN model...")
    knn_model = KNNModel()
    knn_model.train(X_train_scaled, y_train)
    print(f"KNN training completed in {knn_model.training_time:.2f}s")

    # Evaluate model
    knn_metrics = knn_model.evaluate(X_test_scaled, y_test)
    
    # Store results
    store_results(
        model_name="KNN",
        training_time=knn_model.training_time,
        accuracy=knn_metrics['accuracy'],
        auc=knn_metrics['roc_auc'],
        precision=knn_metrics['precision'],
        recall=knn_metrics['recall'],
        f1_score=knn_metrics['f1_score'],
        mcc=knn_metrics['mcc']
    )

    # save model
    knn_model.save_model(os.path.join(MODEL_DIR, "knn_model.pkl"))

    # --------------------------------------------------------------
    # training Gaussian Naive Bayes model
    # --------------------------------------------------------------
    print("Training Gaussian Naive Bayes model...")
    nb_model = NaiveBayesGaussianModel()
    nb_model.train(X_train, y_train)
    print(f"Gaussian Naive Bayes training completed in {nb_model.training_time:.2f}s")

    # Evaluate model
    nb_metrics = nb_model.evaluate(X_test, y_test)
    
    # Store results
    store_results(
        model_name="Gaussian Naive Bayes",
        training_time=nb_model.training_time,
        accuracy=nb_metrics['accuracy'],
        auc=nb_metrics['roc_auc'],
        precision=nb_metrics['precision'],
        recall=nb_metrics['recall'],
        f1_score=nb_metrics['f1_score'],
        mcc=nb_metrics['mcc']
    )

    # save model
    nb_model.save_model(os.path.join(MODEL_DIR, "gaussian_naive_bayes_model.pkl"))

    # --------------------------------------------------------------
    # training Random Forest model
    # --------------------------------------------------------------

    print("Training Random Forest model...")
    rf_model = RandomForestModel(random_state=42)
    rf_model.train(X_train, y_train)
    print(f"Random Forest training completed in {rf_model.training_time:.2f}s")

    # Evaluate model
    rf_metrics = rf_model.evaluate(X_test, y_test)

    # Store results
    store_results(
        model_name="Random Forest",
        training_time=rf_model.training_time,
        accuracy=rf_metrics['accuracy'],
        auc=rf_metrics['roc_auc'],
        precision=rf_metrics['precision'],
        recall=rf_metrics['recall'],
        f1_score=rf_metrics['f1_score'],
        mcc=rf_metrics['mcc']
    )

    # save model
    rf_model.save_model(os.path.join(MODEL_DIR, "random_forest_model.pkl"))

    # --------------------------------------------------------------
    # training XGBoost model
    # --------------------------------------------------------------

    print("Training XGBoost model...")
    xgb_model = XGBoostModel(random_state=42)
    xgb_model.train(X_train, y_train)
    print(f"XGBoost training completed in {xgb_model.training_time:.2f}s")

    # Evaluate model
    xgb_metrics = xgb_model.evaluate(X_test, y_test)

    # Store results
    store_results(
        model_name="XGBoost",
        training_time=xgb_model.training_time,
        accuracy=xgb_metrics['accuracy'],
        auc=xgb_metrics['roc_auc'],
        precision=xgb_metrics['precision'],
        recall=xgb_metrics['recall'],
        f1_score=xgb_metrics['f1_score'],
        mcc=xgb_metrics['mcc']
    )

    # save model
    xgb_model.save_model(os.path.join(MODEL_DIR, "xgboost_model.pkl"))

    # Print final results
    print("\nFinal Results:")
    print(results_df)

    # Save results to CSV
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "model_results.csv"), index=False)

if __name__ == "__main__":
    main()
