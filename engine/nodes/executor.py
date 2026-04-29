from engine.runner import run_pipelines
from pipelines.text.simple import run as simple_run
from pipelines.text.llm_models import run as llm_run
from pipelines.ocr.tesseract_pipeline import run as tess_run
from pipelines.ocr.easyocr_pipeline import run as easy_run
from pipelines.asr.whisper_pipeline import run as whisper_run

def executor_node(state):

    pipeline_map = {
        "simple": simple_run,
        "mistral": lambda x: llm_run("mistral", x),
        "phi3": lambda x: llm_run("phi3", x),
        "tesseract": tess_run,
        "easyocr": easy_run,
        "whisper": whisper_run
    }

    class Dummy:
        pass

    context = Dummy()
    context.input = state["input"]
    context.pipeline_candidates = state["pipeline_candidates"]

    context = run_pipelines(context, pipeline_map)

    state["results"] = context.results

    print(f"[Executor] Ran {len(state['results'])} pipelines")

    return state