from utils.experiment_logger import save_experiment, print_summary_table

def recommender_node(state):
    summary = state.get("summary", {})
    if not summary:
        print("[Recommender] No summary found!")
        return state

    # Select best by score
    best_name = max(summary, key=lambda x: summary[x]["score"])
    best_metrics = summary[best_name]
    
    state["best_pipeline"] = best_name
    state["decision_reason"] = (
        f"Selected {best_name} (Acc: {best_metrics['avg_accuracy']:.2f}, "
        f"Lat: {best_metrics['avg_latency']:.2f}s) based on highest aggregated score."
    )

    print(f"[Recommender] Best: {best_name}")
    
    # Final logging and display
    save_experiment(state.get("scored_results", []), summary)
    print_summary_table(summary)
    
    return state