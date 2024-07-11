import os
import shutil
import yaml


# Function to load the configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Function to copy the configuration file
def copy_config_file(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.copy(source, destination)
