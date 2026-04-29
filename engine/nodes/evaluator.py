from utils.metrics import levenshtein_similarity

def evaluator_node(state):
    results = state.get("raw_results", [])
    task = state.get("task")
    
    scored_results = []
    
    print(f"[Evaluator] Scoring for task: {task.name if task else 'Default'}")

    for r in results:
        raw_output = r.get("output", "")
        gt = r.get("ground_truth", "")
        
        # 1. Direct Score Check (Benchmark APIs)
        if "score" in r:
            acc = r["score"]
            processed_output = raw_output
        else:
            # 2. Post-process
            if task and task.postprocess_fn:
                processed_output = task.postprocess_fn(raw_output)
            else:
                processed_output = raw_output
                
            # 3. Compute Metric
            if task and hasattr(task, "metric_fn"):
                acc = task.metric_fn(processed_output, gt)
            else:
                acc = levenshtein_similarity(processed_output, gt)
            
        # 3. Clamp
        acc = max(0.0, min(acc, 1.0))
        
        # 4. Sample Penalty
        if r.get("error"):
            sample_score = -1000
        else:
            sample_score = acc
            
        r["accuracy"] = acc
        r["sample_score"] = sample_score
        r["processed_output"] = processed_output
        scored_results.append(r)
        
    state["scored_results"] = scored_results
    return state