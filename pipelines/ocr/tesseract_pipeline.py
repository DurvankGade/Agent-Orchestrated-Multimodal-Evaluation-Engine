import pytesseract
from PIL import Image
import time

def run(input_data: str, task_instruction: str) -> dict:
    """
    input_data: path to image
    task_instruction: instruction (ignored by Tesseract but kept for signature)
    """
    start = time.time()

    try:
        img = Image.open(input_data)
        text = pytesseract.image_to_string(img)
        error = None
    except Exception as e:
        text = ""
        error = str(e)

    latency = time.time() - start

    return {
        "pipeline": "tesseract",
        "provider": "tesseract",
        "type": "model",
        "output": text.strip(),
        "latency": latency,
        "cost": 0,
        "cost_type": "simulated",
        "error": error
    }