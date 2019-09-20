from flask import Blueprint
from model.create_keras_model import create

edible_plants = Blueprint('edible_plants', __name__)


@edible_plants.route('/create-model', methods=["GET"])
def create_model():

    # model = create()

    return 'edible plant...or is it?!'
