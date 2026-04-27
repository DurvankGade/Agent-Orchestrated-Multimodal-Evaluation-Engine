import time

def run(input_text):
    start = time.time()

    output = input_text.lower().strip()

    latency = max(time.time() - start, 1e-6) 

    return {
        "pipeline": "simple_text",
        "output": output,
        "latency": latency,
        "cost": 0
    }