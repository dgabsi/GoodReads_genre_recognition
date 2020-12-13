import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input,GlobalAveragePooling2D,BatchNormalization, Conv2D,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow_addons.metrics import F1Score


class ConvGoodReads(object):
        """
        Class that creates Convolutional  newral networks to predicting the goodreads dataset genres.
        Able to build two types of models:
        Convolutional  based on transfer learning
        and another simple convolutional neural net.
        """
        def __init__(self, data_path, batch_size,num_classes, image_size):
                """
                Initalization of interal attributes: data_path -path to diretory containg images
                batch_size
                number of classes to predict
                image size requested
                """
                self.data_path=data_path
                self.transfer_learning_model=None
                self.model=None
                self.batch_size=batch_size
                self.num_classes=num_classes
                self.image_size=image_size

        def build_transfer_learning_model(self,top_config=[256,64], dropout_pct=0.2):
                """
                Creates a model bases on transfer learning with types being base on resnet model
                Args:
                top_config- number of units in the top layers after the transfer model
                dropout_pct- dropout percentage for dropout layer
                """

                inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
                preprocessed_inputs = preprocess_input(inputs)
                #prepare transfer learning model
                self.main_model = ResNet50(weights="imagenet", include_top=False, pooling="max")
                main_model_output = self.main_model(preprocessed_inputs)

                #creating top model .After the tranfer learning model
                x = Flatten()(main_model_output)
                x = Dense(top_config[0], activation="relu")(x)
                x = Dropout(dropout_pct)(x)
                x = BatchNormalization()(x)
                x = Dense(top_config[1], activation="relu")(x)
                outputs = Dense(self.num_classes, activation="softmax")(x)
                self.transfer_learning_model = tf.keras.Model(inputs, outputs)



        def fit_transfer_learning_model(self,train_len,val_len, num_epochs=10, learning_rate=0.001,patience=5):
                """
                Fits the transfer learning convolutional model
                Args:
                num_epochs- Limit on the number of epochs
                train_len- number of rows in train dataset(used to calc the train steps)
                val_len -number of rows in train dataset(used to calc the train steps)
                learning_rate
                patience -For early stopping .Number of epochs to wait if early stopping condition is met
                """

                self.transfer_train_generator = ImageDataGenerator()
                self.transfer_val_generator = ImageDataGenerator()


                # Preparing the flow from images directories
                self.transfer_train_generator = self.transfer_train_generator.flow_from_directory(
                        os.path.join(self.data_path, "images-train"),
                        target_size=(self.image_size, self.image_size),
                        batch_size=self.batch_size,
                        class_mode='categorical')
                self.transfer_val_generator = self.transfer_val_generator.flow_from_directory(
                        os.path.join(self.data_path, "images-val"),
                        target_size=(self.image_size, self.image_size),
                        batch_size=self.batch_size,
                        class_mode='categorical')

                self.main_model.trainable = True

                train_steps_per_epoch = int(train_len / self.batch_size) + 1
                val_steps_per_epoch = int(val_len / self.batch_size) + 1

                self.transfer_learning_optimizer = Adam(learning_rate)
                #compile the model again after changing the trainable params and before feeting
                self.transfer_learning_model.compile(optimizer=self.transfer_learning_optimizer, loss="categorical_crossentropy",metrics=[F1Score(num_classes=self.num_classes, average="weighted"), "accuracy"])
                #add an early stopping callback
                early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
                history = self.transfer_learning_model.fit(self.transfer_train_generator, steps_per_epoch=train_steps_per_epoch,epochs=num_epochs, validation_steps=val_steps_per_epoch, callbacks=[early_stop_callback],validation_data=self.transfer_val_generator)

                return history

        def build_model(self, conv2d_config=[32, 128, 64], kernel_size=5, dense_config=[128, 64], pool_size=2):
                """
                Creates a simple convolutional model
                Args:
                conv2d_config- number of filters in the each conv2d layer
                kernel_size - kernel size (kernel_size*kernel_size for conv2 layers)
                dense_config -number of units in the each dense layer
                pool_size -pool size for pooling layers
                """
                self.model=models.Sequential([
                        Conv2D(kernel_size=kernel_size, filters=conv2d_config[0], activation='relu', padding="same", input_shape=[self.image_size,self.image_size,3]),
                        MaxPooling2D(pool_size=pool_size),
                        BatchNormalization(),
                        Conv2D(kernel_size=kernel_size, filters=conv2d_config[1], activation='relu', padding="same"),
                        MaxPooling2D(pool_size=pool_size),
                        BatchNormalization(),
                        Conv2D(kernel_size=kernel_size, filters=conv2d_config[2], activation='relu', padding="same"),
                        MaxPooling2D(pool_size=pool_size),
                        BatchNormalization(),
                        Flatten(),
                        Dense(units=dense_config[0],activation='relu' ),
                        Dense(units=dense_config[1], activation='relu'),
                        Dense(units=self.num_classes, activation='softmax')])



        def fit_model(self, num_epochs,train_len,val_len, learning_rate, patience=5):
                """
                Fits the simple convolutional
                Args:
                num_epochs- Limit on the number of epochs
                train_len- number of rows in train dataset(used to calc the train steps)
                val_len -number of rows in train dataset(used to calc the train steps)
                learning_rate
                patience -For early stopping .Number of epochs to wait if early stopping condition is met
                 """

                self.train_generator = ImageDataGenerator(rescale=1. / 255)
                self.val_generator = ImageDataGenerator(rescale=1. / 255)

                # Preparing the flow from images directories
                self.train_generator = self.train_generator.flow_from_directory(
                        os.path.join(self.data_path, "images-train"),
                        target_size=(self.image_size, self.image_size),
                        batch_size=self.batch_size,
                        class_mode='categorical')
                self.val_generator = self.val_generator.flow_from_directory(
                        os.path.join(self.data_path, "images-val"),
                        target_size=(self.image_size, self.image_size),
                        batch_size=self.batch_size,
                        class_mode='categorical')

                train_steps_per_epoch=int(train_len/self.batch_size)+1
                val_steps_per_epoch = int(val_len / self.batch_size) + 1
                print(learning_rate)
                self.optimizer = Adam(learning_rate)
                # compile the model again after changing the trainable params and before feeting
                self.model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=[F1Score(num_classes=self.num_classes, average="weighted"), "accuracy"])
                #Add an early stopping callback
                early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
                history = self.model.fit(self.train_generator, steps_per_epoch=train_steps_per_epoch,
                                                           epochs=num_epochs, validation_steps=val_steps_per_epoch,
                                                           validation_data=self.val_generator, callbacks=[early_stop_callback])

                return history

        def save_model(self,model_type,models_path,file_name):
                """
                Saves the model to h5 file
                model_type- either transfer_learning' or 'simple_conv'
                """
                if model_type == 'transfer_learning':
                        self.transfer_learning_model.save(os.path.join(models_path, file_name + ".h5"))
                elif model_type == 'simple_conv':
                        self.model.save(os.path.join(models_path, file_name + ".h5"))

        def load_from_saved(self,model_type,models_path,file_name):
                """
                Loads the model from file- model and weights.
                model_type- either transfer_learning' or 'simple_conv'
                """
                if model_type=='transfer_learning':
                        self.transfer_learning_model = models.load_model(os.path.join(models_path, file_name+".h5"))
                elif model_type=='simple_conv':
                        self.model = models.load_model(os.path.join(models_path, file_name+".h5"))

        def predict(self, model_type, data_path,directory_name):
                """
                Predict test data according to transfer learning model
                """
                if model_type == 'transfer_learning':
                        test_generator = ImageDataGenerator()
                else:
                        test_generator = ImageDataGenerator(rescale=1. / 255)

                test_generator = test_generator.flow_from_directory( os.path.join(data_path, directory_name),
                        target_size=(self.image_size, self.image_size),
                        batch_size=self.batch_size,
                        class_mode='categorical')

                if model_type == 'transfer_learning':
                        y_predict=self.transfer_learning_model.predict(test_generator)
                else:
                        y_predict=self.model.predict(test_generator)
                return y_predict

        def evaluate(self, model_type, data_path, directory_name):
                """
                Evaluates metrics on requested data.
                Return a dictionary containing all metrics
                """

                #prepare requested data generators
                if model_type == 'transfer_learning':
                        test_generator = ImageDataGenerator()
                else:
                        test_generator = ImageDataGenerator(rescale=1. / 255)


                test_generator = test_generator.flow_from_directory(os.path.join(data_path, directory_name),
                                                                    target_size=(self.image_size, self.image_size),
                                                                    batch_size=self.batch_size,
                                                                    class_mode='categorical')

                if model_type == 'transfer_learning':
                        scores_dict = self.transfer_learning_model.evaluate(test_generator,batch_size=self.batch_size, return_dict=True)
                else:
                        scores_dict = self.model.evaluate(test_generator,batch_size=self.batch_size,return_dict=True)
                return scores_dict










