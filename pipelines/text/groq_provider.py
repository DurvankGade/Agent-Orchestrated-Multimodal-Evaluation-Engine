import time
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"

def run(input_data: str, task_instruction: str) -> dict:
    start = time.time()
    
    prompt = f"{task_instruction}\n\nInput: {input_data}"

    output = ""
    error = None

    try:
        res = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            },
            timeout=10
        )

        if res.status_code != 200:
            error = f"HTTP {res.status_code}: {res.text}"
        else:
            data = res.json()
            if "choices" in data and len(data["choices"]) > 0:
                output = data["choices"][0]["message"]["content"].strip()
            else:
                output = str(data)
                error = "Unexpected response format"

    except Exception as e:
        error = str(e)

    latency = max(time.time() - start, 1e-6)

    return {
        "pipeline": "groq",
        "provider": "groq",
        "type": "api",
        "output": output,
        "latency": latency,
        "cost": 0.0005,
        "cost_type": "simulated",
        "error": error
    }