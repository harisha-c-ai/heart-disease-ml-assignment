import time
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

class BaseModel:
    """
    Base class encapsulating common training and evaluation logic
    for all classification models.
    """
    def __init__(self, model):
        self.model = model
        self.training_time = None

    def train(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        self.training_time = end_time - start_time

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "mcc": matthews_corrcoef(y_test, y_pred)
        }
        return metrics
    
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)