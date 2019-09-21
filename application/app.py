#!flask/bin/python
from flask import Flask, render_template
from flask_cors import CORS
from blueprints import register_blueprints

UPLOAD_FOLDER = 'images/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':

    print('registering blueprints')
    register_blueprints(app)
    print('finished registering blueprints')

    app.run(debug=True, port=5000, use_reloader=False)
