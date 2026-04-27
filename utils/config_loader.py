import yaml

def load_config(path="configs/weights.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)