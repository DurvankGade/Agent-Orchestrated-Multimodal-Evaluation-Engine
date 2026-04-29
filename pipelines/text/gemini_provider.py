import time, os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def run(input_data: str, task_instruction: str) -> dict:
    start = time.time()
    
    prompt = f"{task_instruction}\n\nInput: {input_data}"

    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        output = res.text.strip() if hasattr(res, "text") and res.text else str(res)
        error = None
    except Exception as e:
        output = ""
        error = str(e)

    return {
        "pipeline": "gemini",
        "provider": "gemini",
        "type": "api",
        "output": output,
        "latency": max(time.time() - start, 1e-6),
        "cost": 0.002,
        "cost_type": "simulated",
        "error": error
    }