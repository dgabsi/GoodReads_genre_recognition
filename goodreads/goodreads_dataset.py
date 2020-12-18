import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler,PowerTransformer,LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer

#import dump_datasets_to_pickle, load_datasets_from_pickle


class GoodreadsDataset(object):
    """
    This class is a service class for preprocessing the data of the Goodreads dataset.
    It includes functions for preliminary preprocessing and further preprocessing for specific inference tasks.
    It is also in charge of splitting the dataset to train,validation and test datasets.
    It prepares both the tabular and images datasets.
    After the preprocessing it holds as attributes the scalers and transformers that are used on the train set and stores them for further use on the val and test datasets.
    At certain stages the processed data is stored to files, which will later enables constructing the class from the preprocessed files and by that avoid repeating preprocessing.
    One of the main reasons of forming a class for preporcessing task is to encapsulate all these procedures and also a class
    holds the different scalers and imputers used in preporcessing task for later use for other data.
    """
    def __init__(self, preprocessed_config, image_source_path, source_images_list, data_path):
        """
        Constructor.Initilization of the preporcess configuration(order of cols requested) and data paths
        Args:
            preprocessed_config- dictionary containtin configuration attributes
            image_source_path-path to directory containing the images of books
            source_image_list-name of file in image_source_path containing inventory of images and their corresponding book id
            data_path- path to data directory. In this directory files will be saved and diretcories for train/val/test images will be created.
        """
        self.data = None
        self.preprocessed_config = preprocessed_config
        self.is_prepared_for_inference = False
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.image_source_path = image_source_path
        self.source_images_list = source_images_list
        self.data_path = data_path
        self.preprocess_types={ "book_id": int, "work_id": int, "title": str, "num_pages": int,"publication_year": int,"description": str,
                           "is_ebook": int,"series": int, "image_url": str,"read_count":int,"text_reviews_count": int,
                           "ratings_count": int, "average_rating": float, "author_id": int,"name": str, "public_year_null": int,
                          "num_pages_null": int,"authors_ratings_count": int,"author_average_rating": float,"genre": str}


    @classmethod
    def load_from_split(cls, preprocessed_config, image_source_path, source_images_list, data_path):
        """
        Constructor that constructs a goodreads dataset from a preprared data split. The splitted data is not yet ready for inference.
        Args:
        preprocessed_config- dictionary containtin configuration attributes
        image_source_path-path to directory containing the images of books
        source_image_list-name of file in image_source_path containing inventory of images and their corresponding book id
        data_path- path to data directory. In this the files are saved and diretcories for train/val/test images exist.
        """
        grd = cls(preprocessed_config, image_source_path, source_images_list, data_path)

        from . import dump_datasets_to_pickle, load_datasets_from_pickle
        #load the preporcessed dataset from a preprepared pickle file
        grd.data = load_datasets_from_pickle(data_path, 'data.pkl')

        # load the preporcessed train dataset from pickle files
        data = load_datasets_from_pickle(data_path, 'train.pkl')
        # reshape to 2dims
        grd.X_train = data[:, :-1]
        grd.y_train = data[:, -1].reshape((-1, 1))

        # load the preporcessed validation dataset from pickle files
        data = load_datasets_from_pickle(data_path, 'val.pkl')
        # reshape to 2dims
        grd.X_val = data[:, :-1]
        grd.y_val = data[:, -1].reshape((-1, 1))

        # load the preporcessed test dataset from pickle files
        data = load_datasets_from_pickle(data_path, 'test.pkl')
        #reshape to 2dims
        grd.X_test = data[:, :-1]
        grd.y_test = data[:, -1].reshape((-1, 1))

        print(f"Loaded train dataset containing: {len(grd.X_train)} samples")
        print(f"Loaded validation dataset containing: {len(grd.X_val)} samples")
        print(f"Loaded test dataset containing: {len(grd.X_test)} samples")


        return grd


    def preprocess(self, data):
        """
        Preprocess books data.
        Includes handling nulls in publication year ands num pages and genres.
        Extracting the main genre (the target) of the book from possible genres.
        Assigning 1/0 to columns is_ebook and series
        Grouping authors data and merging it to books data
        Ordering of the columns to final structure according to config params
        Args: data-books input data (DataFrame)
              preprocessed_config-dictionary containing necessary configuration data such requested order of column  in output data
        Return value: preprocessed books data (DataFrame)
        """

        self.data = data
        # Handling publication year
        #only take books that were published between 1500 and 2021 or null which will be imputed to median value publication year
        self.data = self.data.loc[
                    (data["publication_year"].isnull() | self.data["publication_year"].between(1500, 2021)), :]
        self.data.loc[:, ["public_year_null"]] = 0
        #before imputing the null values to median add a feature that remembers that this was originally null
        self.data.loc[self.data["publication_year"].isnull(), "public_year_null"] = 1
        self.data["publication_year"].fillna(self.data["publication_year"].median(), inplace=True)

        # Handling number of pages
        # changes zero values of pages numbers to nan before imputing them to median
        self.data.loc[self.data["num_pages"]==0,"num_pages"] = np.nan
        self.data.loc[:, ["num_pages_null"]] = 0
        # before imputing the null values to median add a feature that remembers that this was originally null
        self.data.loc[self.data["num_pages"].isnull(), "num_pages_null"] = 1
        self.data["num_pages"].fillna(self.data["num_pages"].median(), inplace=True)

        genres = self.preprocessed_config["genres_col"]
        self.data.set_index("book_id", inplace=True)
        genres_df = self.data[genres]
        genres_df["genre"] = genres_df.idxmax(axis="columns")
        self.data = self.data.join(genres_df["genre"])
        self.data.reset_index(inplace=True)

        self.data.loc[(self.data["genre"].isnull() & self.data["fiction"] > 0), ["genre"]] = "fiction"

        self.data.dropna(subset=["genre"], inplace=True)
        genres = genres + ["fiction"]
        self.data.drop(columns=genres, inplace=True)

        # Handling genres
        # genres=["fiction","poetry","children","fantasy, paranormal","history, historical fiction, biography",
        #  "comics, graphic","non-fiction","mystery, thriller, crime","young-adult","romance"]
        # bigger_1 = (lambda x : 1 if x>1 else 0)
        # data.loc[:,genres]=data.loc[:,genres].applymap(bigger_1)

        is_ebook = (lambda x: 1 if x else 0)
        self.data.loc[:, "is_ebook"] = self.data.loc[:, "is_ebook"].apply(is_ebook)

        ser = lambda x: 1 if x != '[]' else 0
        self.data["series"] = self.data["series"].apply(ser)


        self.data.drop(columns=["Unnamed: 0"], inplace=True)



        authors = self.data[["author_id", "ratings_count", "average_rating"]]
        authors["sum_rating"] = authors["average_rating"] * authors["ratings_count"]
        authors = authors.groupby(["author_id"])["ratings_count", "sum_rating"].sum()
        authors["author_average_rating"] = authors["sum_rating"] / authors["ratings_count"]
        authors.drop(columns=["sum_rating"], inplace=True)
        authors.rename(columns={"ratings_count": "authors_ratings_count"}, inplace=True)

        self.data = self.data.merge(authors, how="left", left_on="author_id", right_on="author_id")

        # data["rating_count_without"]=data["ratings_count_y"]-data["ratings_count_x"]
        # data["sum_counts_book"]=data["average_rating_x"]*data["ratings_count_x"]
        # data["sum_rating"]=data["sum_rating"]-data["sum_counts_book"]
        # data["author_rating_without"]=data["sum_rating"]/data["rating_count_without"]
        # data.drop(columns=["sum_rating","rating_count_without","sum_counts_book","ratings_count_y","average_rating_y"],inplace=True)
        # data.rename({"rating_count_x":"ratings_count","average_rating_x":"average_rating"}, inplace=True)

        # data.drop(columns=["book_id", "work_id","title", "name","description", "Unnamed: 0", "Unnamed: 0.1","image_url"],inplace=True)


        self.data = self.data[self.preprocessed_config["preprocess_col_order"]]

        self.data= self.data.astype(self.preprocess_types)

        # turn columns to correct data types
        from . import dump_datasets_to_pickle, load_datasets_from_pickle
        dump_datasets_to_pickle(self.data_path, self.data, "data.pkl")

        return self.data

    def prepare_images_genres(self, dataset_dir_name, X, y, book_id_index):
        """
        prepare images files belonging to the features data in the requested path.
        Copying images files from images source directory to the requested directory. Only images that belong to X dataset will be copied .
        Creating sub folders according to labels(y) and assigning images to the correct subfolder according to its label.

        Args: image_source_path- path to source images files directory
          source_images_files_df- Dataframe of books and their images file
          data_path- path to data directory. In this path images datasets directories should be created.
          dataset_dir_name-name of the dataset directory that should be created under data_path
          X- dataset of features (2d array)
          y- labels array
          book_id_index- index of the book id column in X features dataset. Images should be copied to subfolder only for the book id's in X dataset.
        Return value: None
        """

        dataset_path = os.path.join(self.data_path, dataset_dir_name)

        print("dataset path ", dataset_path)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.mkdir(dataset_path)

        source_images_files_df = pd.read_csv(os.path.join(self.image_source_path, self.source_images_list))
        for ind in range(len(X)):
            label = y[ind]
            book_id = X[ind, book_id_index]
            full_book_filename = \
            (source_images_files_df.loc[source_images_files_df["book_id"] == str(book_id), "full_filename"]).values[0]

            label_path = os.path.join(dataset_path, label)
            if not os.path.exists(label_path):
                os.mkdir(label_path)
            image_file = full_book_filename
            shutil.copyfile(os.path.join(self.image_source_path, image_file), os.path.join(label_path, image_file))

    def prepare_train_test_split(self, test_pct, val_pct, seed):
        """
        prepare train, validation and test datasets from the full data.
        Handle preparation for both array and images files data. Images are orgnised in folders according to the dataset type and label.
        Copying images files from images source directory to the requested directory. Only images that belong to features data will be copied.

        Args: X-books input data (two dimensional array)
          y- label data (one dimension array)
          test_pct- test percentage from input data
          val_pct- validation percentage from train data
          image_source_path- path to source images files directory
          source_images_list- name of file containing list of books and their images file(The file is located in image_source_path)
          data_path- path to data directory. In this path images datasets directories should be created.
          preprocessed_col- list of columns of the input data
        Return value: X_train, y_train, X_val, y_val, X_test, y_test -features array and label array for each of the datasets.     """


        if self.data is None:
            raise Exception('Error: Must preporcess data first')

        features = list(self.data.columns)
        features.remove("genre")

        #split to features and labels
        X = self.data[features].values
        y = self.data["genre"].values

        #split the dataset to train, validation and test datasets
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(X, y, test_size=test_pct, shuffle=True,
                                                                              stratify=y, random_state=seed)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val,
                                                                              test_size=val_pct, shuffle=True,
                                                                              stratify=y_train_val,random_state=seed)

        print(f"Loaded train dataset containing: {len(self.X_train)} samples")
        print(f"Loaded validation dataset containing: {len(self.X_val)} samples")
        print(f"Loaded test dataset containing: {len(self.X_test)} samples")

        images_files_df = pd.read_csv(os.path.join(self.image_source_path, self.source_images_list))


        self.prepare_images_genres('images-train', self.X_train, self.y_train,
                                   self.preprocessed_config["preprocess_col_order"].index("book_id"))
        self.prepare_images_genres('images-val', self.X_val, self.y_val,
                                   self.preprocessed_config["preprocess_col_order"].index("book_id"))
        self.prepare_images_genres('images-test', self.X_test, self.y_test,
                                   self.preprocessed_config["preprocess_col_order"].index("book_id"))

        print(f"All images prepared in folders")

        #increase the dimention of labels to be 2 dim array
        self.y_train = self.y_train[:, np.newaxis]
        self.y_val = self.y_val[:, np.newaxis]
        self.y_test = self.y_test[:, np.newaxis]

        from . import dump_datasets_to_pickle, load_datasets_from_pickle
        dump_datasets_to_pickle(self.data_path, np.hstack((self.X_train, self.y_train)), "train.pkl")
        dump_datasets_to_pickle(self.data_path, np.hstack((self.X_val, self.y_val)), "val.pkl")
        dump_datasets_to_pickle(self.data_path, np.hstack((self.X_test, self.y_test)), "test.pkl")
        print(f"Dataset splits saved as pickle files")

        return None

    def prepare_data_for_inference(self, add_text_features=True, descr_maxwords=1000, title_maxwords=1000, drop_ids=True, drop_author_id=True,target_hot_encoding=False):

        self.is_prepared_for_inference = True

        #Change the datasets back to Dataframes to be easier to work with
        self.X_train= pd.DataFrame(self.X_train, columns=self.preprocessed_config["preprocess_col_order"][:-1])
        self.X_val = pd.DataFrame(self.X_val, columns=self.preprocessed_config["preprocess_col_order"][:-1])
        self.X_test = pd.DataFrame(self.X_test, columns=self.preprocessed_config["preprocess_col_order"][:-1])

        #change types of values
        preprocess_types_without_label=self.preprocess_types.copy()
        preprocess_types_without_label.pop("genre")
        self.X_train=self.X_train.astype(preprocess_types_without_label)
        self.X_val = self.X_val.astype(preprocess_types_without_label)
        self.X_test = self.X_test.astype(preprocess_types_without_label)

        #drop not needed features
        self.X_train.drop(columns=["image_url", "name"], inplace=True)
        self.X_val.drop(columns=["image_url", "name"], inplace=True)
        self.X_test.drop(columns=["image_url", "name"], inplace=True)

        #we save the books id's for later before dropping them (in order to be able to recognise later which book is it.
        self.train_book_ids=self.X_train[["book_id"]]
        self.val_book_ids = self.X_val[["book_id"]]
        self.test_book_ids = self.X_test[["book_id"]]
        if drop_ids:
            self.X_train.drop(columns=["book_id", "work_id"] , inplace=True)
            self.X_val.drop(columns=["book_id", "work_id"], inplace=True)
            self.X_test.drop(columns=["book_id", "work_id"], inplace=True)

        #drop author id if requested.
        if drop_author_id:
            self.X_train.drop(columns=["author_id"], inplace=True)
            self.X_val.drop(columns=["author_id"], inplace=True)
            self.X_test.drop(columns=["author_id"], inplace=True)

        #change target values to one hot encoding if requested using OneHotEncoder
        if target_hot_encoding:
            genre_ohe = LabelBinarizer()
            self.y_train = genre_ohe.fit_transform(self.y_train)
            self.label_classes = genre_ohe.classes_
            self.y_val = genre_ohe.transform(self.y_val)
            self.y_test = genre_ohe.transform(self.y_test)

        # Normalising ratings. Since it has a values between 1-5 we use min max scaler.
        self.books_r_scaler = MinMaxScaler()
        self.X_train["average_rating"] = self.books_r_scaler.fit_transform(self.X_train["average_rating"].values[:, np.newaxis])
        self.X_val["average_rating"]=self.books_r_scaler.transform(self.X_val["average_rating"].values[:, np.newaxis])
        self.X_test["average_rating"] = self.books_r_scaler.transform(self.X_test["average_rating"].values[:, np.newaxis])

        # Normalising authors ratings. Since it has a values between 1-5 we use min max scaler.
        self.author_r_scaler = MinMaxScaler()
        self.X_train["author_average_rating"] = self.author_r_scaler.fit_transform(self.X_train["author_average_rating"].values[:, np.newaxis])
        self.X_val["author_average_rating"] = self.author_r_scaler.transform(self.X_val["author_average_rating"].values[:, np.newaxis])
        self.X_test["author_average_rating"] = self.author_r_scaler.transform(self.X_test["author_average_rating"].values[:, np.newaxis])

        # Normalising publication year. Since it has a values between 1500 and 2021 we use min max scaler.
        self.publ_year_scaler = MinMaxScaler()
        self.X_train["publication_year"] = self.publ_year_scaler.fit_transform(self.X_train["publication_year"].values[:, np.newaxis])
        self.X_val["publication_year"] = self.publ_year_scaler.transform(self.X_val["publication_year"].values[:, np.newaxis])
        self.X_test["publication_year"] = self.publ_year_scaler.transform(self.X_test["publication_year"].values[:, np.newaxis])

        # Normalising number of pages. We use power transformer since it handles outliers well and number of pages is positive
        self.page_count_scaler = PowerTransformer(method='box-cox')
        self.X_train["num_pages"] = self.page_count_scaler.fit_transform(self.X_train["num_pages"].values[:, np.newaxis])
        self.X_val["num_pages"] = self.page_count_scaler.transform(self.X_val["num_pages"].values[:, np.newaxis])
        self.X_test["num_pages"] = self.page_count_scaler.transform(self.X_test["num_pages"].values[:, np.newaxis])

        # Normalising ratings count. We use power transformer since it handles outliers well and ratings count is positive
        self.ratings_count_scaler = PowerTransformer(method='box-cox')
        self.X_train["ratings_count"] = self.ratings_count_scaler.fit_transform(self.X_train["ratings_count"].values[:, np.newaxis])
        self.X_val["ratings_count"] = self.ratings_count_scaler.transform(self.X_val["ratings_count"].values[:, np.newaxis])
        self.X_test["ratings_count"] = self.ratings_count_scaler.transform(self.X_test["ratings_count"].values[:, np.newaxis])

        # Normalising read count. We use power transformer since it handles outliers well and read count is positive
        self.books_read_count_scaler = PowerTransformer(method='box-cox')
        self.X_train["read_count"] = self.books_read_count_scaler.fit_transform(self.X_train["read_count"].values[:, np.newaxis])
        self.X_val["read_count"] = self.books_read_count_scaler.transform(self.X_val["read_count"].values[:, np.newaxis])
        self.X_test["read_count"] = self.books_read_count_scaler.transform(self.X_test["read_count"].values[:, np.newaxis])

        # Normalising text reviews count. We use power transformer since it handles outliers well and text reviews count is positive
        self.books_text_r_count_scaler = PowerTransformer(method='box-cox')
        self.X_train["text_reviews_count"] = self.books_text_r_count_scaler.fit_transform(self.X_train["text_reviews_count"].values[:, np.newaxis])
        self.X_val["text_reviews_count"] = self.books_text_r_count_scaler.transform(self.X_val["text_reviews_count"].values[:, np.newaxis])
        self.X_test["text_reviews_count"] = self.books_text_r_count_scaler.transform( self.X_test["text_reviews_count"].values[:, np.newaxis])

        # Normalising authors ratings count. We use power transformer since it handles outliers well and authors ratings is positive
        self.authors_rate_count_scaler = PowerTransformer(method='box-cox')
        self.X_train["authors_ratings_count"] = self.authors_rate_count_scaler.fit_transform(self.X_train["authors_ratings_count"].values[:, np.newaxis])
        self.X_val["authors_ratings_count"] = self.authors_rate_count_scaler.transform(self.X_val["authors_ratings_count"].values[:, np.newaxis])
        self.X_test["authors_ratings_count"] = self.authors_rate_count_scaler.transform(self.X_test["authors_ratings_count"].values[:, np.newaxis])

        #adding text features - tfifd normalization for both description and title
        if add_text_features:
            #Handle description
            # prepare tfidf
            self.desc_tfidf = TfidfVectorizer(max_features=descr_maxwords, stop_words='english')
            self.desc_tfidf.fit(self.X_train["description"])
            self.tfidf_desc_words=['tfidf_desc_' + word for word in self.desc_tfidf.get_feature_names()]
            self.X_train[self.tfidf_desc_words]=self.desc_tfidf.transform(self.X_train["description"]).toarray()
            self.X_val[self.tfidf_desc_words] = self.desc_tfidf.transform(self.X_val["description"]).toarray()
            self.X_test[self.tfidf_desc_words] = self.desc_tfidf.transform(self.X_test["description"]).toarray()

            # prepare bag of words
            self.desc_bow = CountVectorizer(max_features=descr_maxwords, stop_words='english')
            self.desc_bow.fit(self.X_train["description"])
            self.bow_desc_words = ['bow_desc_' + word for word in self.desc_bow.get_feature_names()]
            self.X_train[self.bow_desc_words] = self.desc_bow.transform(self.X_train["description"]).toarray()
            self.X_val[self.bow_desc_words] = self.desc_bow.transform(self.X_val["description"]).toarray()
            self.X_test[self.bow_desc_words] = self.desc_bow.transform(self.X_test["description"]).toarray()

            self.X_train.drop(columns=["description"], inplace=True)
            self.X_val.drop(columns=["description"], inplace=True)
            self.X_test.drop(columns=["description"], inplace=True)

            # Handle title
            #prepare tfidf
            self.title_tfidf = CountVectorizer(max_features=title_maxwords, stop_words='english')
            self.title_tfidf.fit(self.X_train["title"])
            self.title_words = ['tfidf_title_' + word for word in self.title_tfidf.get_feature_names()]
            self.X_train[self.title_words] = self.title_tfidf.transform(self.X_train["title"]).toarray()
            self.X_val[self.title_words] = self.title_tfidf.transform(self.X_val["title"]).toarray()
            self.X_test[self.title_words] = self.title_tfidf.transform(self.X_test["title"]).toarray()

            # prepare bag of words
            self.title_bow = TfidfVectorizer(max_features=title_maxwords, stop_words='english')
            self.title_bow.fit(self.X_train["title"])
            self.bow_title_words= ['bow_title_' + word for word in self.title_bow.get_feature_names()]
            self.X_train[self.bow_title_words] = self.title_bow.transform(self.X_train["title"]).toarray()
            self.X_val[self.bow_title_words] = self.title_bow.transform(self.X_val["title"]).toarray()
            self.X_test[self.bow_title_words] = self.title_bow.transform(self.X_test["title"]).toarray()

            self.X_train.drop(columns=["title"], inplace=True)
            self.X_val.drop(columns=["title"], inplace=True)
            self.X_test.drop(columns=["title"], inplace=True)


        #saves final columns order
        self.inference_col_names=self.X_train.columns

        print("All datasets prepared for inference")

        return None




