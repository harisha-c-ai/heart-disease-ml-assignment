from xgboost import XGBClassifier
from model.base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        model = XGBClassifier(**kwargs)
        super().__init__(model)