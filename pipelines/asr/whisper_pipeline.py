import whisper
import time

# load once
model = whisper.load_model("base")

def run(input_data: str, task_instruction: str) -> dict:
    start = time.time()

    try:
        result = model.transcribe(input_data)
        output = result["text"].strip()
        error = None
    except Exception as e:
        output = ""
        error = str(e)

    latency = max(time.time() - start, 1e-6)

    return {
        "pipeline": "whisper",
        "provider": "openai",
        "type": "model",
        "output": output,
        "latency": latency,
        "cost": 0,
        "cost_type": "simulated",
        "error": error
    }