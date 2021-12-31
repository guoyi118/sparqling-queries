from flask import Blueprint
from flask_restful import Api
from service import Service

api_blueprint = Blueprint("api", __name__)
api = Api(api_blueprint)

api.add_resource(Service, '/QDMR') # replace with your api name