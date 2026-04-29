from utils.config_loader import load_config
from utils.experiment_logger import save_experiment

def aggregator_node(state):
    scored_results = state.get("scored_results", [])
    config = load_config()
    
    w_acc = config.get("accuracy_weight", 0.7)
    w_lat = config.get("latency_weight", 0.2)
    w_cost = config.get("cost_weight", 0.1)
    max_lat = config.get("max_latency_threshold", 5.0)
    max_cost = config.get("max_cost_threshold", 0.05)
    fail_penalty = config.get("failure_penalty", 10)
    
    pipelines = {}
    for r in scored_results:
        p_name = r["pipeline"]
        p_type = r.get("type", "unknown")
        if p_name not in pipelines:
            pipelines[p_name] = {"accs": [], "lats": [], "costs": [], "fails": 0, "type": p_type}
        
        pipelines[p_name]["accs"].append(r["accuracy"])
        pipelines[p_name]["lats"].append(r["latency"])
        pipelines[p_name]["costs"].append(r["cost"])
        if r.get("error"):
            pipelines[p_name]["fails"] += 1
            
    summary = {}
    for p_name, data in pipelines.items():
        n = len(data["accs"])
        avg_acc = sum(data["accs"]) / n if n > 0 else 0
        avg_lat = sum(data["lats"]) / n if n > 0 else 0
        total_cost = sum(data["costs"])
        fails = data["fails"]
        
        acc_clamped = max(0.0, min(avg_acc, 1.0))
        lat_norm = min(avg_lat / max_lat, 1.0)
        cost_norm = min(total_cost / max_cost, 1.0)
        
        # Mandatory Score Formula
        score = (acc_clamped * w_acc) - (lat_norm * w_lat) - (cost_norm * w_cost) - (fails * fail_penalty)
        
        summary[p_name] = {
            "type": data["type"],
            "avg_accuracy": avg_acc,
            "avg_latency": avg_lat,
            "total_cost": total_cost,
            "failure_count": fails,
            "score": score
        }
        
    state["summary"] = summary
    save_experiment(scored_results, summary) # Save to logs
    return state
