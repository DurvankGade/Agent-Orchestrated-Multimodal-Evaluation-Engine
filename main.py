import os
import argparse
from engine.core.dataset import Dataset, TEXT_SAMPLES, OCR_SAMPLES, ASR_SAMPLES
from engine.core.task import Task
from engine.graph.agent_graph import build_graph
from utils.metrics import semantic_similarity, levenshtein_similarity
from utils.experiment_logger import print_summary_table

# --- Reusable Task Definitions ---
def get_task(task_name: str) -> Task:
    tasks = {
        "summarization": Task(
            name="summarization",
            instruction="Summarize the following text in one sentence: {input}",
            metric_fn=semantic_similarity
        ),
        "question_answering": Task(
            name="question_answering",
            instruction="Answer the following question based on the context: {input}",
            metric_fn=semantic_similarity
        ),
        "ocr_extraction": Task(
            name="ocr_extraction",
            instruction="Extract all text from this image exactly.",
            metric_fn=levenshtein_similarity
        ),
        "speech_transcription": Task(
            name="speech_transcription",
            instruction="Transcribe this audio clip exactly.",
            metric_fn=levenshtein_similarity
        )
    }
    return tasks.get(task_name)

def get_dataset(dataset_name: str) -> Dataset:
    datasets = {
        "text_prod": Dataset("text_prod", "text", TEXT_SAMPLES),
        "ocr_prod": Dataset("ocr_prod", "image", OCR_SAMPLES),
        "asr_prod": Dataset("asr_prod", "audio", ASR_SAMPLES)
    }
    return datasets.get(dataset_name)

def run_evaluation(task_name: str, mode: str, dataset_name: str):
    task = get_task(task_name)
    dataset = get_dataset(dataset_name)
    
    if not task or not dataset:
        print(f"Error: Invalid task '{task_name}' or dataset '{dataset_name}'")
        return None

    print("\n" + "="*80)
    print(f"DATASET: {dataset.name} (n={len(dataset)})")
    print(f"MODE: {mode}")
    print(f"TASK: {task.name}")
    print("="*80)

    state = {
        "dataset": dataset.samples,
        "modality": dataset.modality,
        "mode": mode,
        "task": task
    }

    graph = build_graph()
    final_state = graph.invoke(state)

    summary = final_state.get("summary", {})
    best_pipeline = final_state.get("best_pipeline")
    decision_reason = final_state.get("decision_reason")

    print_summary_table(summary)
    
    print(f"\nBEST: {best_pipeline}")
    print(f"\nDecision:\n{decision_reason}")

    return final_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Evaluation Engine CLI")
    parser.add_argument("--task", type=str, default="summarization", choices=["summarization", "question_answering", "ocr_extraction", "speech_transcription"])
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "production"])
    parser.add_argument("--dataset", type=str, default="text_prod", choices=["text_prod", "ocr_prod", "asr_prod"])

    args = parser.parse_args()

    run_evaluation(args.task, args.mode, args.dataset)