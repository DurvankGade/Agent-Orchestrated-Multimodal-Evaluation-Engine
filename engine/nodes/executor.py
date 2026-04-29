from pipelines.text.gemini_provider import run as gemini
from pipelines.text.groq_provider import run as groq
from pipelines.text.llm_models import mistral, phi3, llama3
from pipelines.text.benchmark_api import run as benchmark_api
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
    "benchmark_api": benchmark_api,
    "tesseract": tess,
    "easyocr": easy,
    "whisper": whisper
}

def executor_node(state):
    dataset = state.get("dataset", [])
    candidates = state.get("pipeline_candidates", [])
    task = state.get("task")
    
    all_results = []

    print(f"[Executor] Running {len(candidates)} pipelines for Task: {task.name if task else 'None'}")

    for sample in dataset:
        sample_input = sample["input"]
        sample_gt = sample["ground_truth"]
        
        # Determine task instruction
        instruction = task.instruction if task else "Process the input."
            
        for p_name in candidates:
            try:
                # All pipelines now follow: run(input_data, task_instruction)
                res = MAP[p_name](sample_input, instruction)
                
                res["ground_truth"] = sample_gt
                res["sample_input"] = sample_input
                all_results.append(res)
            except Exception as e:
                all_results.append({
                    "pipeline": p_name,
                    "provider": p_name,
                    "output": "",
                    "latency": 5.0,
                    "cost": 0,
                    "cost_type": "simulated",
                    "error": str(e),
                    "ground_truth": sample_gt,
                    "sample_input": sample_input
                })

    state["raw_results"] = all_results
    return state