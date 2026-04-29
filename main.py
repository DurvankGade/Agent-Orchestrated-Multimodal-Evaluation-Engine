from engine.state.context import Context
from engine.runner import run_pipelines

from pipelines.text.simple import run as simple_run
from pipelines.text.llm_models import run as llm_run

from evaluators.composite import score


def run_text_example():
    context = Context("Hello World", modality="text")

    context.pipeline_candidates = ["simple", "mistral", "phi3"]

    pipeline_map = {
        "simple": simple_run,
        "mistral": lambda x: llm_run("mistral", x),
        "phi3": lambda x: llm_run("phi3", x),
    }

    context = execute_and_evaluate(context, pipeline_map, "hello world")

    print("\nTEXT RESULTS:")
    for r in context.results:
        print(r)

    print("\nBEST TEXT:")
    print(context.best)

from pipelines.ocr.tesseract_pipeline import run as tess_run
from pipelines.ocr.easyocr_pipeline import run as easy_run

def run_ocr_example():
    context = Context("data/samples/test1.png", modality="image")

    context.pipeline_candidates = ["tesseract", "easyocr"]

    pipeline_map = {
        "tesseract": tess_run,
        "easyocr": easy_run
    }

    ground_truth = "If poison expires is it more poisonous or no longer poisonous"

    context = execute_and_evaluate(context, pipeline_map, ground_truth)

    print("\nOCR RESULTS:")
    for r in context.results:
        print(r)

    print("\nBEST OCR:")
    print(context.best)

def execute_and_evaluate(context, pipeline_map, ground_truth):
    context = run_pipelines(context, pipeline_map)

    context.results = [score(r, ground_truth) for r in context.results]

    context.best = max(context.results, key=lambda x: x["score"])

    return context

from engine.graph.agent_graph import build_graph

def run_audio_example():
    graph = build_graph()

    state = {
        "input": r"data\samples\dont_forget_the_jacket.wav",
        "modality": "audio",
        "ground_truth": "Don't forget the jacket"  # adjust to your audio
    }

    result = graph.invoke(state)

    print("\nAUDIO RESULT:")
    print(result)

def run_agent_system():

    graph = build_graph()

    state = {
        "input": r"data/samples/dont_forget_the_jacket.wav",
        "modality": "audio",
        "ground_truth": "Don't forget the jacket"
    }

    result = graph.invoke(state)

    print("\nFINAL STATE:")
    print(result)

if __name__ == "__main__":

    from utils.config_loader import load_config
    print("\nCONFIG:", load_config())
    run_agent_system()

