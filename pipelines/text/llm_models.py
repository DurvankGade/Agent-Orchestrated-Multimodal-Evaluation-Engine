import time
from langchain_ollama import OllamaLLM

# Initialize models once (important for performance)
MODELS = {
    "mistral": OllamaLLM(model="mistral", keep_alive=0),
    "phi3": OllamaLLM(model="phi3", keep_alive=0),
    "llama3": OllamaLLM(model="llama3", keep_alive=0)
    # add more here later
}

def build_prompt(input_text: str) -> str:
    return f"""
Convert the following text to lowercase.
Return ONLY the transformed text.
Do not explain anything.

Input: {input_text}
Output:
"""

def run(model_name: str, input_text: str) -> dict:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found")

    llm = MODELS[model_name]

    start = time.time()

    prompt = build_prompt(input_text)
    response = llm.invoke(prompt)

    latency = max(time.time() - start, 1e-6)

    return {
        "pipeline": f"{model_name}_text",
        "output": response.strip(),
        "latency": latency,
        "cost": 0
    }