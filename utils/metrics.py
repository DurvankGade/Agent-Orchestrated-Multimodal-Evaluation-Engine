from difflib import SequenceMatcher
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Load a lightweight model for semantic similarity
# 'all-MiniLM-L6-v2' is very fast and small (~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize(text: str) -> str:
    if not text: return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def levenshtein_similarity(pred: str, gt: str) -> float:
    p = normalize(pred)
    g = normalize(gt)
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    return max(0.0, min(SequenceMatcher(None, p, g).ratio(), 1.0))

def semantic_similarity(pred: str, gt: str) -> float:
    """Uses embeddings to compute cosine similarity"""
    p = normalize(pred)
    g = normalize(gt)
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    
    try:
        embeddings = model.encode([p, g])
        # cosine similarity is 1 - cosine distance
        sim = 1 - cosine(embeddings[0], embeddings[1])
        return max(0.0, min(float(sim), 1.0))
    except Exception as e:
        print(f"[Metric Error] Semantic similarity failed: {e}")
        return levenshtein_similarity(pred, gt) # fallback

def exact_match(pred: str, gt: str) -> float:
    return 1.0 if normalize(pred) == normalize(gt) else 0.0
