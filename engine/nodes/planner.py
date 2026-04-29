from langchain_ollama import OllamaLLM
import json

llm = OllamaLLM(model="phi3")

AVAILABLE_PIPELINES = [
    "simple",
    "phi3",
    "mistral",
    "llama3",
    "gemini",
    "groq",
    "benchmark_api",
    "tesseract",
    "easyocr",
    "whisper"
]

def planner_node(state):
    modality = state["modality"]
    mode = state.get("mode", "production")
    
    print(f"\n[Planner] Mode: {mode} | Modality: {modality}")

    # 1. Strict Benchmark Mode
    if mode == "benchmark":
        selected = ["gemini", "groq", "benchmark_api"]
        state["pipeline_candidates"] = selected
        print(f"[Planner] Strict Benchmark Selection: {selected}")
        return state

    # 2. Hard constraints for non-text in Production
    if modality == "image":
        selected = ["tesseract", "easyocr"]
        state["pipeline_candidates"] = selected
        return state

    if modality == "audio":
        selected = ["whisper"]
        state["pipeline_candidates"] = selected
        return state

    # 3. LLM-driven decision for text in Production
    prompt = f"""
You are an AI system planner.
Modality: {modality}
Available pipelines: {AVAILABLE_PIPELINES}

Guidelines:
- Choose best pipelines for the current modality.
- Prefer Mistral/Phi3 for local text tasks.
- Prefer Gemini/Groq for complex API tasks.

Return ONLY a JSON list of pipelines.
"""

    try:
        response = llm.invoke(prompt)
        start = response.find("[")
        end = response.find("]") + 1
        selected = json.loads(response[start:end])
        selected = [p for p in selected if p in AVAILABLE_PIPELINES]
    except:
        selected = ["simple", "phi3"]

    state["pipeline_candidates"] = selected
    print(f"[Planner LLM Selected]: {selected}")
    return state