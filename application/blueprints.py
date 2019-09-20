# Import every blueprint file
from views import sentiment


def register_blueprints(app):
    """
    Adds all blueprint objects into the app.
    """
    app.register_blueprint(sentiment.sentiment)

    # All done!
    app.logger.info("Blueprints registered")
