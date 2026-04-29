from langchain_ollama import OllamaLLM
import json

llm = OllamaLLM(model="phi3")

AVAILABLE_PIPELINES = [
    "simple",
    "phi3",
    "mistral",
    "llama3",
    "gemini",
    "grok",
    "tesseract",
    "easyocr",
    "whisper"
]

def planner_node(state):
    modality = state["modality"]
    user_input = state["input"]
    mode = state.get("mode", "production")

    # 1. Hard constraints (never let LLM break these)
    if modality == "image":
        selected = ["tesseract", "easyocr"]
        state["pipeline_candidates"] = selected
        print("[Planner] Image pipelines:", selected)
        return state

    if modality == "audio":
        selected = ["whisper"]
        state["pipeline_candidates"] = selected
        print("[Planner] Audio pipelines:", selected)
        return state

    # 2. Benchmark mode (explicit comparison)
    if mode == "benchmark":
        selected = ["gemini", "grok"]
        state["pipeline_candidates"] = selected
        print("[Planner] Benchmark pipelines:", selected)
        return state

    # 3. LLM-driven decision for text
    prompt = f"""
You are an AI system planner.

Input: "{user_input}"
Modality: {modality}

Available pipelines:
{AVAILABLE_PIPELINES}

Guidelines:
- Use simple, phi3 or mistral for short/simple inputs
- Use mistral or API models for complex inputs
- Prefer lower latency when accuracy is similar
- Avoid unnecessary expensive API calls

Return ONLY a JSON list of pipelines.
Example:
["simple", "phi3"]
"""

    try:
        response = llm.invoke(prompt)

        # Extract JSON safely
        start = response.find("[")
        end = response.find("]") + 1
        selected = json.loads(response[start:end])

        # Validate pipelines
        selected = [p for p in selected if p in AVAILABLE_PIPELINES]

        if not selected:
            selected = ["simple", "phi3"]

    except Exception as e:
        print("[Planner Error]", e)
        selected = ["simple", "phi3"]

    state["pipeline_candidates"] = selected
    print("[Planner LLM Selected]", selected)

    return state