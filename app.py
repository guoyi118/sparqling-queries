from flask import Flask
from flask_cors import CORS
from api import api_blueprint
import os

class Config:
    DEBUG = True
    UPLOADS_DEFAULT_DEST = os.getcwd()
    UPLOADS_DEFAULT_URL = 'http://localhost:6000/'


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_object(Config)
    app.register_blueprint(api_blueprint)
    return app