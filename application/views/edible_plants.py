from flask import Blueprint, request, render_template, url_for
import os
from model.create_keras_model import create_model_and_save_weights, predict_images_against_model

import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from werkzeug.utils import secure_filename

from keras.preprocessing import image

edible_plants = Blueprint('edible_plants', __name__)

WEIGHTS_FILE = 'edible_weights_v1.h5'


@edible_plants.route('/create-model', methods=["GET"])
def create_model():

    # Create our model
    print('Creating the model')
    weights_created = create_model_and_save_weights(WEIGHTS_FILE)

    return 'Weights file saved in /static/weights' if weights_created else 'Not created'


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@edible_plants.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, '../static/uploads', secure_filename(f.filename))
        f.save(file_path)

        # Create upload image array
        uploaded_image_path = os.path.join(basepath, '../static/uploads', f.filename)
        uploaded_image_array = [uploaded_image_path]

        # Make prediction and return result
        return render_template('index.html',
                               predicted_text=predict_images_against_model(uploaded_image_array,
                                                                           os.path.join(basepath, '../model',
                                                                                        WEIGHTS_FILE)),
                               sent_image=url_for('static', filename='uploads/' + f.filename))

    return 'An error occurred'
