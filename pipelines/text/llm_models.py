import time
from langchain_ollama import OllamaLLM

# Initialize models once
MODELS = {
    "mistral": OllamaLLM(model="mistral", keep_alive=0),
    "phi3": OllamaLLM(model="phi3", keep_alive=0),
    "llama3": OllamaLLM(model="llama3", keep_alive=0)
}

def run_model(model_name: str, input_data: str, task_instruction: str) -> dict:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found")

    llm = MODELS[model_name]
    start = time.time()

    prompt = f"{task_instruction}\n\nInput: {input_data}"
    
    try:
        response = llm.invoke(prompt)
        output = response.strip()
        error = None
    except Exception as e:
        output = ""
        error = str(e)

    latency = max(time.time() - start, 1e-6)

    return {
        "pipeline": model_name,
        "provider": "ollama",
        "type": "model",
        "output": output,
        "latency": latency,
        "cost": 0,
        "cost_type": "simulated",
        "error": error
    }

# Helper wrappers for specific models
def mistral(input_data, task_instruction): return run_model("mistral", input_data, task_instruction)
def phi3(input_data, task_instruction): return run_model("phi3", input_data, task_instruction)
def llama3(input_data, task_instruction): return run_model("llama3", input_data, task_instruction)