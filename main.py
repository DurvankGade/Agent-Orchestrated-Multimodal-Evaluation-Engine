import os
from engine.core.dataset import Dataset, TEXT_SAMPLES, OCR_SAMPLES, ASR_SAMPLES
from engine.core.task import Task
from engine.graph.agent_graph import build_graph
from utils.metrics import semantic_similarity, levenshtein_similarity
from utils.experiment_logger import print_summary_table

def run_benchmark(dataset: Dataset, task: Task, mode: str = "benchmark"):
    print("\n" + "="*80)
    print(f"DATASET: {dataset.name} | MODE: {mode} | TASK: {task.name}")
    print("="*80)

    state = {
        "dataset": dataset.samples,
        "modality": dataset.modality,
        "mode": mode,
        "task": task
    }

    graph = build_graph()
    final_state = graph.invoke(state)

    print("\n[CRITIC ANALYSIS]")
    for pipeline, analysis in final_state.get("analysis", {}).items():
        print(f"\n{analysis}")
    
    print("\n[SUMMARY TABLE]")
    print_summary_table(final_state.get("summary", {}))
    
    print(f"\nFINAL RECOMMENDATION: {final_state.get('best_pipeline')}")

if __name__ == "__main__":
    # 1. Summarization Task (Text, Semantic)
    sum_task = Task(
        name="summarization",
        instruction="Summarize the following text in one sentence: {input}",
        metric_fn=semantic_similarity
    )
    
    # 2. QA Task (Text, Semantic)
    qa_task = Task(
        name="question_answering",
        instruction="Answer the following question based on the context: {input}",
        metric_fn=semantic_similarity
    )
    
    # 3. OCR Task (Image, Levenshtein)
    ocr_task = Task(
        name="ocr_extraction",
        instruction="Extract all text from this image exactly.",
        metric_fn=levenshtein_similarity
    )
    
    # 4. ASR Task (Audio, Levenshtein)
    asr_task = Task(
        name="speech_transcription",
        instruction="Transcribe this audio clip exactly.",
        metric_fn=levenshtein_similarity
    )

    # DATASETS
    text_ds = Dataset("text_prod", "text", TEXT_SAMPLES)
    ocr_ds = Dataset("ocr_prod", "image", OCR_SAMPLES)
    
    # RUN SUMMARIZATION BENCHMARK
   # run_benchmark(text_ds, sum_task, mode="benchmark")

    # RUN QA PRODUCTION
    # run_benchmark(text_ds, qa_task, mode="production")
    
    # RUN OCR BENCHMARK
    run_benchmark(ocr_ds, ocr_task, mode="production")

    # RUN ASR PRODUCTION
    #run_benchmark(asr_ds, asr_task, mode="production")