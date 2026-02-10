from sklearn.tree import DecisionTreeClassifier
from model.base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self, **kwargs):
        model = DecisionTreeClassifier(**kwargs)
        super().__init__(model)
