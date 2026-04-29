import time
import random

def run(input_data: str, task_instruction: str) -> dict:
    """
    Simulated Benchmark API.
    Returns a score directly instead of just output.
    """
    start = time.time()
    
    # Simulate API call
    time.sleep(0.5)
    
    # Simulation: 70-95% accuracy
    sim_score = random.uniform(0.7, 0.95)
    
    return {
        "pipeline": "benchmark_api",
        "provider": "BenchmarkBot",
        "type": "benchmark_api",
        "output": f"Simulated evaluation of: {input_data[:20]}...",
        "score": sim_score, # Direct evaluation score
        "latency": time.time() - start,
        "cost": 0.005,
        "cost_type": "simulated",
        "error": None
    }
