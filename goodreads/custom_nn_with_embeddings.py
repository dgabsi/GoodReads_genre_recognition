import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input,GlobalAveragePooling2D,BatchNormalization,Embedding,GlobalAveragePooling1D,Concatenate,Conv1D, Input,MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
import spacy
import gensim
from gensim.models.word2vec import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow_addons.metrics import F1Score
nltk.download('punkt')
nltk.download('stopwords')

class EmbeddingNNGoodreads(object):
    """
    Class that creates Neural network based on embeddings for predicting the goodreads dataset genres.
    Title, Description and author features are replaced by embeddings
    """
    def __init__(self,batch_size,num_classes=10,embedding_dim=300):
        """
        Initialization of internal attributes
        """
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.title_tokenizer=None
        self.description_tokenizer=None
        self.author_encode = None
        self.pretrained_emb_type=None
        self.title_embeddings=None
        self.desc_embeddings=None

    def remove_stopword(self, string_list):
        """
        Removing english stop words from the each string in lists.
        This is a service function to be used on text before training on model
        """

        stop_word_cleaned_sentences=[]
        stop_words = set(stopwords.words('english'))
        for string_value in string_list:
            string_word_tokens = word_tokenize(string_value)
            cleaned_words = [word for word in string_word_tokens if not word.lower() in stop_words]
            stop_word_cleaned_sentences.append(' '.join(cleaned_words))
        return stop_word_cleaned_sentences

    def train_word2vec_vectors(self,description_array,window_size_desc, title_array,window_size_title, seed ):
        """
        train word2vec embeddings using the gensim package- fot both description and title
        """
        description_word_sequence = [text_to_word_sequence(text) for text in description_array]
        self.description_word2vec_model = Word2Vec(description_word_sequence, window=window_size_desc, size=self.embedding_dim,
                                              min_count=5, iter=100, seed=seed)
        title_word_sequence = [text_to_word_sequence(text) for text in title_array]
        self.title_word2vec_model = Word2Vec(title_word_sequence, window=window_size_title, size=self.embedding_dim, min_count=3, iter=100,seed=seed)

    def prepare_preprocessing_voc_layers(self, title_array,num_voc_title, description_array,num_voc_description, author_array,max_len_title,max_len_desc):
        """
         Prepare Text vectorization layers and embeddings layers to be used later in network
        """
        #Prepare description text vectorization layer. The layer handles tokenization (including handling punctuation) and converting to int sequences.
        self.description_tokenizer = TextVectorization(max_tokens=(num_voc_description+2), output_sequence_length=max_len_desc, name="Tokenizer_description")
        desc_text_ds = tf.data.Dataset.from_tensor_slices(description_array).batch(self.batch_size)
        self.description_tokenizer.adapt(desc_text_ds)

        # Prepare title text vectorization layer. The layer handles tokenization (including handling punctuation) and converting to int sequences.
        self.title_tokenizer = TextVectorization(max_tokens=(num_voc_title+2), output_sequence_length=max_len_title,name="Tokenizer_title")
        title_text_ds = tf.data.Dataset.from_tensor_slices(title_array).batch(self.batch_size)
        self.title_tokenizer.adapt(title_text_ds)

        #convert also author id's to integer sequences to be replaced by embeddings.
        self.author_encode= IntegerLookup(name="Tokenizer_author")
        self.author_encode.adapt(author_array)

        self.desc_voc = self.description_tokenizer.get_vocabulary()
        self.title_voc = self.title_tokenizer.get_vocabulary()
        self.author_voc = self.author_encode.get_vocabulary()


    def prepare_embedding_vectors(self,description_array, title_array,pretrained_emb="spacy",window_size_desc=10,window_size_title=3,seed=42):
        """
        Prepare embeddings- based on pretrained spacy or train of description/title features.
        Creates embedding array the include embedding for each vocabulary word-for description and title
        """

        #Either use pretrained embeddings downloaded from spacty or trained word2vec embedding on our data
        self.pretrained_emb_type=pretrained_emb
        if self.pretrained_emb_type=='spacy':
            spacy_embeddings=spacy.load("en_core_web_md")
        else:
            self.train_word2vec_vectors(description_array, window_size_desc, title_array, window_size_title,seed)


        # Prepare embedding for descriptions. We create an array where each row corresponds to the embedding vector for a token in our vocabulary.
        self.desc_embeddings= np.random.rand(len(self.desc_voc), self.embedding_dim)
        for ind, word in enumerate(self.desc_voc):
            if self.pretrained_emb_type=='spacy':
                embedding_vector=spacy_embeddings(word).vector
            else:
                embedding_vector = (self.description_word2vec_model[word] if word in self.description_word2vec_model.wv.vocab.keys() else None)
            if embedding_vector is not None:
                self.desc_embeddings[ind] = embedding_vector

        # Prepare embedding for descriptions. We create an array where each row corresponds to the embedding vector for a token in our vocabulary.
        self.title_embeddings = np.random.rand(len(self.title_voc), self.embedding_dim)
        for ind, word in enumerate(self.title_voc):
            if self.pretrained_emb_type=='spacy':
                embedding_vector=spacy_embeddings(word).vector
            else:
                embedding_vector = (self.title_word2vec_model[word] if word in self.title_word2vec_model.wv.vocab.keys() else None)
            if embedding_vector is not None:
                self.title_embeddings[ind] = embedding_vector




    def build_model(self,others_number_features,dense_config=[128,64], kernel_size_desc=5, kernel_size_title=3, dropout_rate=0.1, conv_filters_desc=128, conv_filters_title=16):
        """
        Building the model. Created from four inputs:
        others_number_features- number of features(other then description, title,author)
        dense_config- units for dense layer at the top of the network
        kernel_size_desc-kernel size form conv1d layer for description
        kernel_size_title -kernel size form conv1d layer for title
        dropout_rate- rate od dropout layers
        conv_filters_desc - number of filters in convolutional layer from description
        conv_filters_title - number of filters in convolutional layer from title
        """

        # Description input
        desc_input = Input(shape=(1,), dtype=tf.string, name='desc_input')
        vect_desc = self.description_tokenizer(desc_input)
        emb_desc = Embedding(input_dim=len(self.desc_voc), output_dim=self.embedding_dim,
                                              embeddings_initializer=tf.keras.initializers.Constant(
                                                  self.desc_embeddings.copy()),
                                              trainable=True, name="Embedding_description")(vect_desc)
        # avaraging across the embeddings dim
        desc=Conv1D(conv_filters_desc, kernel_size_desc, activation="relu",dtype=tf.float32,name="Convolution_description")(emb_desc)
        desc = MaxPooling1D(pool_size=3,name="Max_pooling_description")(desc)
        desc = Dropout(dropout_rate)(desc)
        desc=GlobalAveragePooling1D(dtype=tf.float32,name="Global_avg_pooling_description")(desc)

        # Title input
        title_input = Input(shape=(1,), dtype=tf.string, name='title_input')
        vect_title = self.title_tokenizer(title_input)
        emb_title = Embedding(input_dim=len(self.title_voc), output_dim=self.embedding_dim,
                                               embeddings_initializer=tf.keras.initializers.Constant(
                                                   self.title_embeddings.copy()),
                                               trainable=True, name="Embedding_title")(vect_title)
        title = Conv1D(conv_filters_title, kernel_size_title, activation="relu", dtype=tf.float32,name="Convolution_title")(emb_title)
        title = MaxPooling1D(pool_size=3,name="Max_pooling_title")(title)
        title = Dropout(dropout_rate)(title)
        #avaraging across the embeddings dim
        title  = GlobalAveragePooling1D(dtype=tf.float32,name="Global_avg_pooling_title")(title)


        #Author input
        author_input = Input(shape=(1,), dtype=tf.int32, name='author_input')
        vect_author = self.author_encode(author_input)
        author_emb = Embedding(input_dim=len(self.author_voc), output_dim=10, trainable=True,name="Embeddings_author")(vect_author)
        # avaraging across the embeddings dim
        author = GlobalAveragePooling1D(dtype=tf.float32,name="Global_avg_pooling_author")(author_emb)

        # Other numerical input
        other_input = Input(shape=(others_number_features,), dtype=tf.float32, name='other_input')

        #Concatenation of all inputs and final Dense layers
        conc_x = Concatenate(name="Concatenation_layer_all_inputs")([desc, title, author,other_input])
        conc_x = BatchNormalization()(conc_x)
        conc_x = Dense(dense_config[0], activation='relu',dtype=tf.float32)(conc_x)
        conc_x =Dropout(dropout_rate)(conc_x)
        conc_x = Dense(dense_config[1], activation='relu',dtype=tf.float32)(conc_x)
        output = Dense(self.num_classes, activation="softmax",dtype=tf.float32)(conc_x)

        self.model = Model(inputs=[desc_input, title_input, author_input,other_input], outputs=output)



    def fit(self, train_others, val_others, train_desc, val_desc,train_title, val_title,train_author, val_author, y_train,y_val, epochs, patience=5, learning_rate=0.003):
        """
        Fitting the model
        """
        #learning_rate
        optimizer = Adam(learning_rate)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=[F1Score(num_classes=self.num_classes, average="weighted"), "accuracy"])

        #Stopping condition on the validation loss . If it has not decreased for more than patience stop training
        es_callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        history=self.model.fit( {'other_input': train_others, 'desc_input': train_desc, 'title_input': train_title, 'author_input': train_author}, y_train,
            validation_data=({'other_input': val_others,'desc_input': val_desc, 'title_input': val_title, 'author_input': val_author}, y_val), epochs=epochs, callbacks=[es_callback])

        return history

    def save_model(self, models_path,file_name):
        """Saving the model
        """
        self.model.save(os.path.join(models_path, file_name + ".h5"))

    def load_from_saved(self, models_path,file_name):
        """Loading the model from file
        """
        self.model = models.load_model(os.path.join(models_path, file_name + ".h5"))

    def predict(self,other_values, desc_values, title_values, author_values):
        """
        Predicting an already trained model on input parameters
        """
        y_predict = self.model.predict({'other_input': other_values, 'desc_input': desc_values, 'title_input': title_values, 'author_input': author_values})
        return y_predict

    def evaluate(self, other_values, desc_values, title_values, author_values, y_true):
        """
        Evaluates metrics on requested data.
        Return a dictionary containing all metrics
        """

        scores_dict = self.model.evaluate(x=
            {'other_input': other_values, 'desc_input': desc_values, 'title_input': title_values, 'author_input': author_values},y=y_true, batch_size=self.batch_size,
                                                                return_dict=True)


        return scores_dict




