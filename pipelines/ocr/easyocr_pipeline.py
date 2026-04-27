import easyocr
import time

# Initialize once (important)
reader = easyocr.Reader(['en'])

def run(image_path: str) -> dict:
    """
    Deep learning OCR pipeline using EasyOCR
    """

    start = time.time()

    results = reader.readtext(image_path)

    # Extract only text parts
    text = " ".join([r[1] for r in results])

    latency = max(time.time() - start, 1e-6)

    return {
        "pipeline": "easyocr_ocr",
        "output": text.strip(),
        "latency": latency,
        "cost": 0
    }