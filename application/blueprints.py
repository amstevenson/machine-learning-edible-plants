# Import every blueprint file
from views import edible_plants


def register_blueprints(app):
    """
    Adds all blueprint objects into the app.
    """
    app.register_blueprint(edible_plants.edible_plants)

    # All done!
    app.logger.info("Blueprints registered")
