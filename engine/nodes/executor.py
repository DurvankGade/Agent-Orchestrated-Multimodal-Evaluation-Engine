from pipelines.text.gemini_provider import run as gemini
from pipelines.text.groq_provider import run as groq
from pipelines.text.llm_models import mistral, phi3, llama3
from pipelines.ocr.tesseract_pipeline import run as tess
from pipelines.ocr.easyocr_pipeline import run as easy
from pipelines.asr.whisper_pipeline import run as whisper
from pipelines.text.simple import run as simple

MAP = {
    "simple": simple,
    "mistral": mistral,
    "phi3": phi3,
    "llama3": llama3,
    "gemini": gemini,
    "groq": groq,
    "tesseract": tess,
    "easyocr": easy,
    "whisper": whisper
}

def executor_node(state):
    dataset = state.get("dataset", [])
    candidates = state.get("pipeline_candidates", [])
    task = state.get("task")
    
    print(f"[Executor] Running {len(candidates)} pipelines for Task: {task.name if task else 'unknown'}")
    
    results = []
    for sample in dataset:
        input_data = sample["input"]
        gt = sample["ground_truth"]
        
        for p_name in candidates:
            if p_name in MAP:
                try:
                    # Inject task instruction
                    instruction = task.instruction if task else "{input}"
                    res = MAP[p_name](input_data, instruction)
                    res["ground_truth"] = gt
                    res["sample_input"] = input_data
                    results.append(res)
                except Exception as e:
                    results.append({
                        "pipeline": p_name,
                        "error": str(e),
                        "ground_truth": gt,
                        "sample_input": input_data,
                        "latency": 0,
                        "cost": 0,
                        "output": ""
                    })
                    
    state["raw_results"] = results
    return state