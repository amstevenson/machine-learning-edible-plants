from flask import Blueprint, request, render_template, url_for
import os, random
from model.create_keras_model import create_model_and_save_weights, predict_images_against_model
from werkzeug.utils import secure_filename

edible_plants = Blueprint('edible_plants', __name__)

WEIGHTS_FILE = 'edible_weights_v1.h5'


@edible_plants.route('/create-model', methods=["GET"])
def create_model():

    # Create our model
    print('Creating the model')
    weights_created = create_model_and_save_weights(WEIGHTS_FILE)

    return 'Weights file saved in /static/weights' if weights_created else 'Not created'


@edible_plants.route('/get-random-predictions', methods=['GET'])
def show_random_predictions():

    random_edible_plants = []
    random_non_edible_images = []
    test_dataset_path = os.path.join(os.path.dirname(__file__),
                                     '../static/Model_Data/test_dataset')

    # Get five random edible plants
    for i in range(0, 5):
        random_edible_plants.append(
            os.path.join(test_dataset_path + '/edible',
                         random.choice(os.listdir(test_dataset_path + '/edible'))))

    # Get five random non-edible plants/things
    for i in range(0, 5):
        random_non_edible_images.append(
            os.path.join(test_dataset_path + '/non-edible',
                         random.choice(os.listdir(test_dataset_path + '/non-edible'))))

    predictions_edible_plants = {}
    predictions_not_edible = {}

    # Get edible images saved in a key format that matches the img element's src parameter
    for key, value in predict_images_against_model(random_edible_plants,
                                                   os.path.join(os.path.dirname(__file__), '../model',
                                                                WEIGHTS_FILE)).items():
        predictions_edible_plants[url_for('static',
                                          filename='Model_Data/test_dataset/edible/' + key.split('/')[-1])] = value

    # Get non-edible images saved in a key format that matches the img element's src parameter
    for key, value in predict_images_against_model(random_non_edible_images,
                                                   os.path.join(os.path.dirname(__file__), '../model',
                                                                WEIGHTS_FILE)).items():
        predictions_not_edible[url_for('static',
                                       filename='Model_Data/test_dataset/non-edible/' + key.split('/')[-1])] = value

    print('(after) Predictions for edible plants: ', predictions_edible_plants)
    print('(after) Predictions for non_edible images: ', predictions_not_edible)

    return render_template('index.html',
                           edible_predictions=predictions_edible_plants,
                           non_edible_predictions=predictions_not_edible)


@edible_plants.route('/predict', methods=['POST'])
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

        # Returns true if edible, or false if not
        edible_plant = predict_images_against_model(uploaded_image_array,
                                                    os.path.join(basepath, '../model',
                                                                 WEIGHTS_FILE))[uploaded_image_path]

        return_text_prediction = 'The Image you uploaded' + (' is not an edible plant'
                                                             if not edible_plant else ' is an edible plant')

        # Make prediction and return result
        return render_template('index.html',
                               predicted_text=return_text_prediction,
                               sent_image=url_for('static', filename='uploads/' + f.filename))

    return 'An error occurred'
