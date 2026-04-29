import easyocr
import time

# Initialize once
reader = easyocr.Reader(['en'])

def run(input_data: str, task_instruction: str) -> dict:
    start = time.time()

    try:
        results = reader.readtext(input_data)
        text = " ".join([r[1] for r in results])
        error = None
    except Exception as e:
        text = ""
        error = str(e)

    latency = max(time.time() - start, 1e-6)

    return {
        "pipeline": "easyocr",
        "provider": "easyocr",
        "type": "model",
        "output": text.strip(),
        "latency": latency,
        "cost": 0,
        "cost_type": "simulated",
        "error": error
    }