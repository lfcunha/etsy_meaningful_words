import configparser
import os

config = configparser.ConfigParser()
config_path = os.path.dirname(os.path.realpath(__file__)) + '/config.ini'
config.read_file(open(config_path))
