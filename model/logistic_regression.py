from sklearn.linear_model import LogisticRegression
from model.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model)