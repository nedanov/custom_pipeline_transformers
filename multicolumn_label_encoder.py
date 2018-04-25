import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#object takes in a pandas dataframe or a numpy matrix and returns a numpy matrix
class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self,columns = None):
        self.columns = columns #a python list containing the indexes of the colums to be labled

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        #if it is a pandas dataframe, extract the values from it
        if isinstance(X,pd.DataFrame):
            output = X.copy().values
        else:
            output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[:,col] = LabelEncoder().fit_transform(output[:,col])
        else:
            n_columns = X.values.shape[1]
            for col in range(0,n_columns):
                output[:,col] = LabelEncoder().fit_transform(output[:,col])
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)