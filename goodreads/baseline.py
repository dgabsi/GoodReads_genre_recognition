import numpy as np
import pandas as pd


class Baseline(object):
    def __init__(self, preprocessed_cols):
        self.preprocessed_cols = preprocessed_cols

    def fit(self, X, y):
        self.data = pd.DataFrame(np.hstack((X, y)), columns=self.preprocessed_cols)
        self.target_mode = self.data["genre"].mode().values[0]

    def predict(self, X):
        author_id_ind = self.preprocessed_cols.index("author_id")
        # print(self.data.loc[:, ["author_id"]].isin([37450]))
        # print(self.data.head())
        # find_genre_by_author= lambda author_id: 'fiction' if self.data.loc[:,"author_id"].isin([author_id]).empty else self.data.loc[self.data["author_id"]==author_id,"genre"].iloc[0]
        # print(X[:, author_id_ind])
        y_predict = pd.Series(X[:, author_id_ind]).apply(self.find_genre_by_author).values

        print(y_predict)
        return y_predict

    def find_genre_by_author(self, author_id):
        # print(author_id)
        # print(self.data.loc[:,"author_id"].isin([author_id]).empty)
        if author_id in self.data["author_id"].values:
            # print(self.data.loc[self.data["author_id"] == author_id, "genre"].values[0])
            return self.data.loc[self.data["author_id"] == author_id, "genre"].values[0]
        else:
            return self.target_mode
