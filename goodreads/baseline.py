import numpy as np
import pandas as pd


class Baseline(object):
    """
    Baseline model. Based on predicting the former genre of an author(otherwise return genre mode)
    """
    def __init__(self, preprocessed_cols):
        self.preprocessed_cols = preprocessed_cols

    def fit(self, X, y):
        """
        Fitting the model in this case we only means saving the training data and finding the mode
        """

        self.data = pd.DataFrame(np.hstack((X, y)), columns=self.preprocessed_cols)
        self.target_mode = self.data["genre"].mode().values[0]

    def predict(self, X):
        """
        For prediction we we only search for the book author in the train data and if found return the genre of the other book by the same author .
        If not found return genre mode.
        """
        author_id_ind = self.preprocessed_cols.index("author_id")
        y_predict = pd.Series(X[:, author_id_ind]).apply(self.find_genre_by_author).values

        return y_predict

    def find_genre_by_author(self, author_id):
        """
        Service function to find the genre of the first book of a author in the train data. to be used by predict()
        """

        if author_id in self.data["author_id"].values:
            return self.data.loc[self.data["author_id"] == author_id, "genre"].values[0]
        else:
            return self.target_mode
