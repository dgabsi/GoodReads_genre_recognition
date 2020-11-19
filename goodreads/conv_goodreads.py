import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, InceptionV3
import tensorflow.keras.applications.inception_v3.preprocess_input as inception_preprocess_input


class ConvGoodReads(object):
        def __init__(self, data_path, batch_size,num_classes, image_size):
                self.data_path=data_path
                self.transfer_learning_model=None
                self.model=None
                self.batch_size=batch_size
                self.num_classes=num_classes
                self.image_size=image_size

        def build_transfer_learning_model(self,model_type="resnet",top_config=[256,64], learning_rate=1e-2):

                self.transfer_model_type=model_type
                inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
                if model_type=="resnet":
                        self.main_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(self.image_size, self.image_size, 3)))
                        main_model_output = self.main_model(inputs)
                elif model_type=="inception":
                        preporcess = inception_preprocess_input(inputs)
                        self.main_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(self.image_size, self.image_size, 3)))
                        main_model_output=self.main_model(preporcess)

                x = GlobalAveragePooling2D()(main_model_output)
                x = Flatten()(x)
                x = Dense(top_config[0], activation="relu")(x)
                x = BatchNormalization()(x)
                x = Dense(top_config[1], activation="relu")(x)
                # x = layers.Dropout(0.1)(x)
                outputs = Dense(self.num_classes, activation="softmax")(x)
                self.transfer_learning_model = tf.keras.Model(inputs, outputs)
                self.optimizer = Adam(learning_rate)
                self.transfer_learning_model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
                print(self.transfer_learning_model.summary())


        def fit_transfer_learning_model(self,model_type="resnet", num_epochs=10, freeze_main_model=True,learning_rate=0.001):

                if self.transfer_model_type == "resnet":
                        self.transfer_train_generator = ImageDataGenerator(rescale=1. / 255)
                        self.transfer_val_generator = ImageDataGenerator(rescale=1. / 255)
                elif self.transfer_model_type == "inception":
                        self.transfer_train_generator = ImageDataGenerator()
                        self.transfer_val_generator = ImageDataGenerator()

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

                if freeze_main_model:
                        self.main_model.trainable = False
                elif:
                        self.main_model.trainable = True

                self.optimizer = Adam(learning_rate)
                self.transfer_learning_model.compile(optimizer=self.optimizer, loss="categorical_crossentropy",metrics=["accuracy"])
                history = model.fit(self.transfer_train_generator, steps_per_epoch=1700,epochs=num_epochs, validation_steps=410, validation_data=self.transfer_val_generator)

                return history

        def build_model(self, config=[64, 256, 10]):
                pass

        def fit_model(self, num_epochs):
                pass

        def save_model(self,model_type,models_path,file_name):
                if model_type=='transfer_learning':
                        self.transfer_learning_model.save(os.path.join(self.data_path, file_name+".h5"))
                elif model_type=='custom_conv':
                        self.model.save(os.path.join(self.data_path, file_name+".h5"))

        def load_from_saved(self,model_type,models_path,file_name):
                if model_type=='transfer_learning':
                        self.transfer_learning_model = models.load_model(os.path.join(self.data_path, file_name+".h5"))
                elif model_type=='custom_conv':
                        self.model = models.load_model(os.path.join(self.data_path, file_name+".h5"))

        def predict_transfer_learning(self):
                if self.transfer_model_type == "resnet":
                        transfer_test_generator = ImageDataGenerator(rescale=1. / 255)
                elif self.transfer_model_type == "inception":
                        transfer_test_generator = ImageDataGenerator()

                transfer_test_generator = self.transfer_test_generator.flow_from_directory(
                        os.path.join(self.data_path, "images-test"),
                        target_size=(self.image_size, self.image_size),
                        batch_size=self.batch_size,
                        class_mode='categorical')

                y_predict=self.transfer_learning_model.predict(transfer_test_generator)

                return y_predict










