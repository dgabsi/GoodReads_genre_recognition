import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

''''
def preprocess(data, preprocessed_config):
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

    # Handling publication year
    data = data.loc[(data["publication_year"].isnull() | data["publication_year"].between(1500, 2021)), :]
    data.loc[:, ["public_year_null"]] = 0
    data.loc[data["publication_year"].isnull(), "public_year_null"] = 1
    data["publication_year"].fillna(data["publication_year"].median(), inplace=True)
    # Handling number of pages
    data.loc[:, ["num_pages_null"]] = 0
    data.loc[data["num_pages"].isnull(), "num_pages_null"] = 1
    data["num_pages"].fillna(data["num_pages"].median(), inplace=True)

    genres = preprocessed_config["genres_col"]
    data.set_index("book_id", inplace=True)
    genres_df = data[genres]
    genres_df["genre"] = genres_df.idxmax(axis="columns")
    data = data.join(genres_df["genre"])
    data.reset_index(inplace=True)

    data.loc[(data["genre"].isnull() & data["fiction"] > 0), ["genre"]] = "fiction"

    data.dropna(subset=["genre"], inplace=True)
    genres = genres + ["fiction"]
    data.drop(columns=genres, inplace=True)

    # Handling genres
    # genres=["fiction","poetry","children","fantasy, paranormal","history, historical fiction, biography",
    #  "comics, graphic","non-fiction","mystery, thriller, crime","young-adult","romance"]
    # bigger_1 = (lambda x : 1 if x>1 else 0)
    # data.loc[:,genres]=data.loc[:,genres].applymap(bigger_1)

    is_ebook = (lambda x: 1 if x else 0)
    data.loc[:, "is_ebook"] = data.loc[:, "is_ebook"].apply(is_ebook)

    ser = lambda x: 1 if x != '[]' else 0
    data["series"] = data["series"].apply(ser)

    data.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "language_code", "format"], inplace=True)

    # data=pd.get_dummies(data)

    authors = data[["author_id", "ratings_count", "average_rating"]]
    authors["sum_rating"] = authors["average_rating"] * authors["ratings_count"]
    authors = authors.groupby(["author_id"])["ratings_count", "sum_rating"].sum()
    authors["author_average_rating"] = authors["sum_rating"] / authors["ratings_count"]
    authors.drop(columns=["sum_rating"], inplace=True)
    authors.rename(columns={"ratings_count": "authors_ratings_count"}, inplace=True)

    data = data.merge(authors, how="left", left_on="author_id", right_on="author_id")

    # data["rating_count_without"]=data["ratings_count_y"]-data["ratings_count_x"]
    # data["sum_counts_book"]=data["average_rating_x"]*data["ratings_count_x"]
    # data["sum_rating"]=data["sum_rating"]-data["sum_counts_book"]
    # data["author_rating_without"]=data["sum_rating"]/data["rating_count_without"]
    # data.drop(columns=["sum_rating","rating_count_without","sum_counts_book","ratings_count_y","average_rating_y"],inplace=True)
    # data.rename({"rating_count_x":"ratings_count","average_rating_x":"average_rating"}, inplace=True)

    # data.drop(columns=["book_id", "work_id","title", "name","description", "Unnamed: 0", "Unnamed: 0.1","image_url"],inplace=True)

    # data=pd.get_dummies(data)
    data = data[preprocessed_config["preprocess_col_order"]]

    return data


def prepare_images_genres(images_source_path, source_images_files_df, data_path, dataset_dir_name, X, y,book_id_index):
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

    dataset_path = os.path.join(data_path, dataset_dir_name)

    print("dataset path ", dataset_path)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    images_files_df=pd.DataFrame(columns=["full_filename","book_id"])
    count=0
    for filename in os.listdir(source_path):
        images_files_df=images_files_df.append({"full_filename":filename,"book_id":filename.split('.')[0]},ignore_index=True)
        count+=1
        if count==10:
            break
        #print(filename)
        #print(filename.split('.')[0])

    print(images_files_df.head())
    print(len(images_files_df))
    for ind in range(len(X)):
        label = y[ind]
        book_id = X[ind, book_id_index]
        # print(f"book:{book_id}")
        print(book_id)
        # print(images_files_df.loc[images_files_df["book_id"] == str(book_id), "full_filename"])
        full_book_filename = (source_images_files_df.loc[source_images_files_df["book_id"] == str(book_id), "full_filename"]).values[0]
        # print(f"book_filename:{full_book_filename}")

        label_path = os.path.join(dataset_path, label)
        # print("label path ", label_path)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        image_file = full_book_filename
        # print(image_file)
        shutil.copyfile(os.path.join(images_source_path, image_file), os.path.join(label_path, image_file))


def prepare_train_test_split(X, y, test_pct, val_pct, image_source_path, source_images_list, data_path, preprocessed_col):
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


    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_pct, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_pct, shuffle=True,stratify=y_train_val)

    images_files_df = pd.read_csv(os.path.join(image_source_path, source_images_list))
    print(preprocessed_col)
    print(preprocessed_col.index("book_id"))

    #images_files_df = pd.DataFrame(columns=["full_filename", "book_id"])
    #for filename in os.listdir(image_source_path):
    #    images_files_df = images_files_df.append({"full_filename": filename, "book_id": filename.split('.')[0]},ignore_index=True)
    prepare_images_genres(image_source_path, images_files_df, data_path, 'images-train', X_train, y_train, preprocessed_col.index("book_id"))
    prepare_images_genres(image_source_path, images_files_df, data_path, 'images-val', X_val, y_val, preprocessed_col.index("book_id"))
    prepare_images_genres(image_source_path, images_files_df, data_path, 'images-test', X_test, y_test, preprocessed_col.index("book_id"))

    return X_train, y_train, X_val, y_val, X_test, y_test
'''''