#!flask/bin/python
from flask import Flask, render_template
from blueprints import register_blueprints

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':

    print('registering blueprints')
    register_blueprints(app)
    print('finished registering blueprints')

    app.run(debug=True, port=5000, use_reloader=False)