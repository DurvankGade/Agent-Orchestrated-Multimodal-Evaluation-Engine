import pytesseract
from PIL import Image
import time

def run(image_path):
    start = time.time()

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)

    latency = time.time() - start

    return {
        "pipeline": "tesseract_ocr",
        "output": text.strip(),
        "latency": latency,
        "cost": 0
    }