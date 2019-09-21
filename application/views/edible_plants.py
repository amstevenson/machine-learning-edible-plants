from flask import Blueprint, request
import os
from model.create_keras_model import create_model_and_save_weights
from werkzeug.utils import secure_filename

edible_plants = Blueprint('edible_plants', __name__)

model = None
WEIGHTS_FILE = 'edible_weights_v1.h5'


@edible_plants.route('/create-model', methods=["GET"])
def create_model():

    # Create our model
    weights_created = create_model_and_save_weights(WEIGHTS_FILE)

    # Save the weights into the static/weights folder
    #save_dir = '{url_for: "/static/weights/"}'

    return 'Weights file saved in /static/weights' if weights_created else 'Not created'


@edible_plants.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        model.load_weights('location of file')

        #preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        #return result
    return None