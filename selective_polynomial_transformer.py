#Transformer which takes the indeces for certain sets of varibles and creates new version of them which
#are raise to a higher degree. this is useful because the default sklearn tranformer automatically squares and
#interacts everything in the feature space
from sklearn.base import TransformerMixin
import numpy as np

class PolynomialTransformer(TransformerMixin):
     
    def __init__(self, indexes = None, degree=2):
        self.indexes = indexes #a python list containing the indexes of the columns to be squared
        self.degree = degree
        
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.indexes is not None:
            for index in self.indexes:
                powered = output[:,index]**2
                output = np.hstack([output,powered.reshape(1,-1).T])
        else:
            pass
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)