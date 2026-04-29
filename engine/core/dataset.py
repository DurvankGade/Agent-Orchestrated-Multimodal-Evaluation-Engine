from typing import List, Dict, Any

class Dataset:
    def __init__(self, name: str, modality: str, samples: List[Dict[str, str]]):
        self.name = name
        self.modality = modality
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"Dataset(name={self.name}, n={len(self)})"

    def get_batch(self) -> List[Dict[str, str]]:
        return self.samples

# Predefined samples for the 4 mandatory tasks
TEXT_SAMPLES = [
    {
        "input": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It is built on top of LangChain and allows you to create graphs where nodes are functions and edges define the control flow.",
        "ground_truth": "LangGraph is a LangChain-based library for creating stateful multi-agent workflows using graphs.",
        "task": "summarization"
    },
    {
        "input": "The capital of France is Paris. It is known for the Eiffel Tower.",
        "ground_truth": "Paris",
        "task": "question_answering"
    },
    {
        "input": "This is a noisy input with lotssss of typoesss and weird characters !!! @#$%",
        "ground_truth": "this is a noisy input with lots of typos and weird characters",
        "task": "summarization"
    }
]

OCR_SAMPLES = [
    {
        "input": "data/samples/clean_text.png",
        "ground_truth": "Extracted Text: This is SAMPLE TEXT Text is at different regions",
        "task": "ocr_extraction"
    },
    {
        "input": "data/samples/test1.png",
        "ground_truth": "If poison expires, is it more poisonous or no longer poisonous?",
        "task": "ocr_extraction"
    }
]

ASR_SAMPLES = [
    {
        "input": "data/samples/don't_forget_the_jacket.wav",
        "ground_truth": "don't forget the jacket",
        "task": "speech_transcription"
    }
]
