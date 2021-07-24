import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'impossible-to-guess-key'

    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///users.db'

    SQLALCHEMY_TRACK_MODIFICATIONS = False