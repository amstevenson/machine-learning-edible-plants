import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Activation, Flatten, SpatialDropout1D
from keras.optimizers import SGD

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

_max_features = 2000


def get_data_and_clean(filename):
    data = pd.read_csv(os.path.join('static', filename),
                       ',', encoding='ISO-8859-1')[:1000]

    print('after csv loaded')

    # Make the text lowercase
    data['SentimentText'] = data['SentimentText'].apply(lambda x: x.lower())

    # Replace any special characters
    data['SentimentText'] = data['SentimentText'].apply(lambda x: re.sub('[^a-z0-9\s]','',x))

    data.head(5)

    print('after cleaning up tweets')
    return data


def get_tokenised_data(data):
    # tokenisation
    # create a tokenizer that takes the 2000 most common words and Separator for word splitting.
    tokenizer = Tokenizer(num_words=_max_features, split=' ')

    # calculate the frequency of each word in the dataset
    # - fit_on_texts creates the vocabulary index based on word frequency.
    # Every word gets its a unique innteger value and 0 is reserved for padding.
    tokenizer.fit_on_texts(data['SentimentText'].values)

    # text_to-sequence basically takes each word in the text and replaces it with its corresponding integer value.
    x = tokenizer.texts_to_sequences(data['SentimentText'].values)

    # applying padding as the NN can train more efficiently on training samples that are the same size
    print('after tokenisation')
    return pad_sequences(x)


def create_and_compile_model(x):
    # larger lstm dropout + lstm_out = embed_dim + extra layers.
    # Note that embed_dim and lstm_out are hyperparameters,
    # their values are somehow intuitive, can be and must be played with in order
    # to achieve good results.
    embed_dim = 156
    lstm_out = 256

    # A Sequential model is a linear stack of layers. Intialise an empty sequential model to add layers to
    model = Sequential()

    # embedding layer lets the network expand each token into a larger vector
    # Fill in the following parameters: 1. size of our vocabulary,
    # 2. embedding dimesion(embed_dim) - expands to a vector of 128
    # 3. length of input
    model.add(Embedding(_max_features, embed_dim,  input_length=x.shape[1]))

    # SpatialDropout1D takes a parameter of a float between 0 and 1. The fraction of the input units to drop.
    model.add(SpatialDropout1D(0.5))

    # CNN layer gives a higher view of the sequences to the LSTM example: "I loved this friendly service"
    # could be processed as "I love this" "Friendly service" which is two chunks for the LSTM rather than 5 chunks
    # 1. the size of our word embeddings, 2&3. dropouts - resetting a random amount of weights (make it harder for the
    # NN to learn patterns)
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

    # Dense is a regular densely-connected NN layer, it takes in units which is a positive
    # integer, dimensionality of the output space and the Activation function to use.
    model.add(Dense(128, activation="tanh"))
    model.add(Dense(2, activation="softmax"))

    # compiles the model using backend library (TensorFlow), using categorical because our targets are one-hot encoded.
    # adam is an algorithm for first-order gradient-based optimization
    # metrics is a list of metrics to be evaluated by the model during training and testing
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('after creating model')

    return model


def get_train_test_split_data(x, y):
    # Use train_test_split to split arrays or matrices into random train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Print out the shape for X and Y both train and test
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return x_train, x_test, y_train, y_test


def fit_model(model, x_train, y_train, x_test, y_test):
    # The batch size defines the number of samples that will be propagated through the network.
    batch_size = 500

    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_test, y_test))


def create():
    # Get and clean data
    data = get_data_and_clean('train.csv')

    # Get tokenised data
    x = get_tokenised_data(data)
    print(x[:1])

    # Create and compile model
    model = create_and_compile_model(x)

    # Get_dummies converts categorical variables into dummy/indicator variables
    y = pd.get_dummies(data['Sentiment']).values
    x_train, x_test, y_train, y_test = get_train_test_split_data(x, y)

    # Train the model
    fit_model(model, x_train, y_train, x_test, y_test)

    print('after fit, x_test: ', x_test)

    # tokenise sentence and use to predict

    # kaggle, click datasets. Or look for 'google datasets'

    # predicting routes for bin collections
    # minimising food waste

    # Ideas:
    # 1) Mushrooms poisonous or not?
    # 2) Poisonous plants
    # 3) Edible plants (have a dataset for this). Anything not classified = not safe!

    # Can use cat vs dogs notebook for example of where we need to send test data
    # after it has been trained.

    predictions = model.predict_classes(x_test)

    print('predictions: ', predictions)

    return model
