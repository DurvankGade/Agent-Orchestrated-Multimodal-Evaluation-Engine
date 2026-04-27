from .accuracy import accuracy
from utils.config_loader import load_config

config = load_config()

def score(result: dict, ground_truth: str) -> dict:
    acc = accuracy(result["output"], ground_truth)

    latency = result["latency"]
    cost = result["cost"]

    score_value = (
        acc * config["accuracy_weight"]
        - latency * config["latency_weight"] * config["latency_penalty_factor"]
        - cost * config["cost_weight"] * config["cost_penalty_factor"]
    )

    result["accuracy"] = acc
    result["score"] = score_value

    return result