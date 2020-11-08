#Import necessary packages
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')
#plt.style.use('ggplot')
import goodreads as gr

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



def learn():
    try:
        with open('configuration.yml', 'r') as file:
            conf = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print('Error reading the config file')

    books_file = os.path.join(conf["files"]["data_dir"], conf["files"]["books_file"])
    books = pd.read_csv(books_file, na_values=['NA', ''])

    books = gr.preprocess(books, conf['columns_lists'])

    print(books_file)

    print(books[["average_rating", "ratings_count", "text_reviews_count", "authors_ratings_count","author_average_rating"]].corr())

    features = list(books.columns)
    features.remove("genre")
    X = books[features].values
    y = books["genre"].values

    data_path = os.path.abspath(conf["files"]["data_dir"])
    image_source_path = os.path.abspath(conf["files"]["image_source_dir"])
    print(data_path)
    print(image_source_path)
    X_train, y_train, X_val, y_val, X_test, y_test = gr.prepare_train_test_split(X, y, test_pct=0.15, val_pct=0.2,image_source_path=image_source_path,  data_path=data_path, preprocessed_col=conf['columns_lists']["preprocessed_col"])

if __name__ == '__main__':
    learn()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
