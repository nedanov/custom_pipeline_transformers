from sklearn.base import TransformerMixin
import numpy as np

class WeightedOneHotEncoder(TransformerMixin):
     
    def __init__(self, indexes = None, weights=None):
        self.indexes = indexes #a python list containing the indexes of the columns to be squared
        self.weights = weights
        
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        shape = X.shape
        n_rows = shape[0]
        n_cols = shape[1]
        n = 0
        if self.indexes is not None:
            for j,index in enumerate(range(n_cols)):
                data_array = output[:,index]
                if j in self.indexes:
                    unique_vals = np.unique(data_array)
                    ohe_matrix = np.zeros((n_rows,len(unique_vals)))
                    for i, val in np.ndenumerate(data_array):
                        i=i[0]
                        col_loc = np.where(unique_vals==val)[0][0]
                        ohe_matrix[i,col_loc]=1
                    scale = self.weights[n]
                    ohe_matrix = scale*ohe_matrix
                    n+=1
                else:
                    ohe_matrix = data_array.reshape(-1,1)
                if j==0:
                    result = ohe_matrix
                else:
                    result = np.hstack([result,ohe_matrix])
        else:
            pass
        return result.astype(float)

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)