import numpy as np
import pandas as pd


class Baseline(object):
    def __init__(self, preprocessed_cols):
        self.preprocessed_cols=preprocessed_cols

    def fit(self,X,y):
        self.full_df=np.concatenate(X,y,axis=1)
    def predict(self,X):

        y_predict=pd.Series([])
        genre_author=lambda author: self.full_df.loc[self.X["author_id"]==author,"genre"].values[0]
        y_predict=X["author_id"].apply(genre_author)
