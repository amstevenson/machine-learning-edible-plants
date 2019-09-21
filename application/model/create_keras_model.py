import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Activation, Flatten, SpatialDropout1D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential, Model, Input
from keras.layers import Activation, Flatten, Dense, Dropout, ZeroPadding2D, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Average
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

import seaborn as sns
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import os
import math

_max_features = 2000
categories = ['edible', 'non_edible']

IMG_WIDTH = 100
IMG_HEIGHT = 100

TRAIN_PATH = os.path.join('static', 'Model_Data/train_dataset')
TEST_PATH = os.path.join('static', 'Model_Data/test_dataset')


def create_model():

    model = Sequential()

    # First Convolution layer
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Conv2D(32, (3,3), activation ='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    #     model.add(Dropout(0.1))

    # Second Convolution layer
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation ='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    #     model.add(Dropout(0.1))

    # Fully-connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(len(categories), activation='softmax'))

    # # Load weights if provided (used in final model)
    # if weights_path is not None:
    #     model.load_weights(None)

    # Compile using Adam optimizer
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def initial_predictions(weights_filename):

    model = create_model()

    model.load_weights(weights_filename)

    print('\n\n\n')

    # Load a few edible images into an array
    EDIBLE = [TEST_PATH + '/edible/Chickweed_test1.jpg',
              TEST_PATH + '/edible/Blue_vervain_test3.jpg',
              TEST_PATH + '/edible/Joe_pye_weed_test3.jpg',

              TEST_PATH + '/edible/Alfalfa_test4.jpg',
              TEST_PATH + '/edible/Alfalfa_test2.jpg',
              TEST_PATH + '/edible/Alfalfa_test3.jpg',

              TEST_PATH + '/edible/Coneflower_test1.jpg',
              TEST_PATH + '/edible/Coneflower_test3.jpg',
              TEST_PATH + '/edible/Coneflower_test5.jpg',

              TEST_PATH + '/edible/Elderberry_test1.jpg']

    # Load a few non_edible images into an array
    NON_EDIBLE = [TEST_PATH + '/non-edible/143.jpg',
                  TEST_PATH + '/non-edible/Fotolia_34870469_XS.jpg',
                  TEST_PATH + '/non-edible/Rhubarb (2).jpg',

                  TEST_PATH + '/non-edible/16.jpg',
                  TEST_PATH + '/non-edible/27.jpg',
                  TEST_PATH + '/non-edible/108.jpg',

                  TEST_PATH + '/non-edible/154.jpg',
                  TEST_PATH + '/non-edible/198.jpg',
                  TEST_PATH + '/non-edible/199.jpg',

                  TEST_PATH + '/non-edible/193.jpg']

    def test_model(array_of_images, model):
        # Loop through the array of images
        for i in range(0, len(array_of_images)):

            # Load the image and resize it
            img = load_img(array_of_images[i], target_size=(IMG_WIDTH, IMG_HEIGHT))

            # Remove the plot x and y ticks
            #plt.xticks([])
            #plt.yticks([])
            # Show the image
            #plt.imshow(img)
            #plt.show()

            # Do transformations on the image so that it can be input as an argument to
            # the model prediction
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            img_classes = model.predict_classes(img)

            # 1 is non-edible and 0 is edible
            print('Prediction for image: ', array_of_images[i], '(name) was: ', img_classes[0],
                  ' The image is NOT EDIBLE ' if img_classes[0] == 1 else ' The image is EDIBLE')

            #if img_classes[0] == 1:
            #    print("The image above is NOT EDIBLE!")
            #else:
            #    print("The image above IS EDIBLE!")

    print('Testing edible')

    test_model(EDIBLE, model)

    print('\n\n\n')
    print('Testing non edible')

    test_model(NON_EDIBLE, model)

    print('\n\n\n')


def create_and_train_model(train_generator, val_generator, weights_filename):

    batch_size = 16
    train_samples = 4000
    validation_samples = 1500

    # Train the network
    model = create_model()
    model.summary()
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=2,
        validation_data=val_generator,
        validation_steps=validation_samples // batch_size)

    model.save_weights(weights_filename)

    return model


def augment_images(image_locs):

    row_count = len(image_locs.index)
    val_split = 0.25
    train_split = 1 - val_split

    # Split the shuffled image_locs into training and validation dataframes by the proportion given by val_split
    train_image_locs = image_locs[:math.floor(train_split * row_count)]
    val_image_locs = image_locs[-math.ceil(val_split * row_count):]

    print(train_image_locs.shape)
    print(val_image_locs.shape)

    # Create image data generator
    train_datagen = ImageDataGenerator(
        rotation_range=40,       # values in degree, range of random rotation
        width_shift_range=0.2,   # shift_range radomly shifts the image horizontally or vertically
        height_shift_range=0.2,
        rescale=1./255,          # normalises the values as 255 is too large for our model to process
        shear_range=0.2,         # randomly applies shear trims
        zoom_range=0.2,          # randomly zooms in on image
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_image_locs,
        directory=TRAIN_PATH,
        x_col='file_loc',
        target_size=(IMG_WIDTH, IMG_HEIGHT)
    )

    val_generator = test_datagen.flow_from_dataframe(
        val_image_locs,
        directory=TRAIN_PATH,
        x_col='file_loc',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        directory=TEST_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        class_mode=None,
        batch_size=1,
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def create_and_return_dataset():

    # Look at all pictures within file path
    folders = os.listdir(TRAIN_PATH)

    print(folders)

    images = []

    for folder in folders:
        if folder != ".DS_Store":

            # first column is class, second column is filename, third column is image address relative to TRAIN_PATH
            files = os.listdir(TRAIN_PATH + '/' + folder)
            images += [(folder, file, folder + '/' + file) for file in files]
            image_locs = pd.DataFrame(images, columns=('class', 'filename', 'file_loc'))

    # Shuffle images, so we have a mix of edible and non-edible. So that they are not in order.
    image_locs = image_locs.sample(frac=1)

    return image_locs


def create_model_and_save_weights(weights_filename):

    # Create and return dataset
    image_locs = create_and_return_dataset()

    print('Finished creating dataset')

    # Augmenting the images
    train_generator, val_generator, test_generator = augment_images(image_locs)

    print('Finished augmenting images')

    # Training the model; use Sam's example to determine how to do this
    create_and_train_model(train_generator, val_generator, weights_filename)

    print('Finished training model')

    # Make initial predictions
    print('Finished making initial predictions for model')
    initial_predictions('edible_weights_v1.h5')

    return True


if __name__ == '__main__':
    # For checking if this all works! Never do this!
    create_model_and_save_weights('edible_weights_v1.h5')

    # initial_predictions('edible_weights_v1.h5')


