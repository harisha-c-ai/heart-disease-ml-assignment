from sklearn.naive_bayes import GaussianNB
from model.base_model import BaseModel

class NaiveBayesGaussianModel(BaseModel):
    def __init__(self, **kwargs):
        model = GaussianNB(**kwargs)
        super().__init__(model)