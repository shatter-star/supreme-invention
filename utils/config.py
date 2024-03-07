import os
import yaml

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config