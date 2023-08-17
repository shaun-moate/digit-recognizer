import yaml
from box import ConfigBox

def load_params(params_file: str):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params
