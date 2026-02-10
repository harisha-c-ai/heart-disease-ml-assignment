from sklearn.neighbors import KNeighborsClassifier
from model.base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, **kwargs):
        model = KNeighborsClassifier(**kwargs)
        super().__init__(model)