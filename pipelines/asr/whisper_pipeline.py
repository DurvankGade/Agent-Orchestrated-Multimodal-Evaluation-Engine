import whisper
import time

# load once (important)
model = whisper.load_model("base")

def run(audio_path: str) -> dict:
    start = time.time()

    result = model.transcribe(audio_path)

    latency = max(time.time() - start, 1e-6)

    return {
        "pipeline": "whisper_asr",
        "output": result["text"].strip(),
        "latency": latency,
        "cost": 0
    }