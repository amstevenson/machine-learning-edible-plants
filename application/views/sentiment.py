from flask import Blueprint
from model.create_keras_model import create

sentiment = Blueprint('sentiment', __name__)


@sentiment.route('/create-model', methods=["GET"])
def create_model():

    model = create()

    return 'came back ok'
