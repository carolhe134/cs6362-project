import importlib.util
import sys
import os

# Path to selfiesvae
selfiesvae_path = os.path.abspath(os.path.join(os.getcwd(), "../../selfiesvae"))
module_name = "selfiesvae"

# Load the module dynamically
spec = importlib.util.spec_from_file_location(module_name, os.path.join(selfiesvae_path, "__init__.py"))
selfiesvae = importlib.util.module_from_spec(spec)
sys.modules[module_name] = selfiesvae
spec.loader.exec_module(selfiesvae)
from selfiesvae.tune_hp import tune_hyperparameters

def main():
    tune_hyperparameters("settings_50k_hp_tuning.yml", 25)

if __name__ == '__main__':
    main()