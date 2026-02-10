from sklearn.ensemble import RandomForestClassifier
from model.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        model = RandomForestClassifier(**kwargs)
        super().__init__(model)