import json
import os

def save_experiment(results: list, summary: dict, output_dir: str = "logs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save raw results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Save aggregated summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

def print_summary_table(summary: dict):
    print("\n" + "="*95)
    print(f"{'Pipeline':<15} | {'Type':<15} | {'Accuracy':<10} | {'Latency':<10} | {'Cost':<10} | {'Failures':<10} | {'Score':<10}")
    print("-" * 95)
    
    # Sort by score descending
    sorted_items = sorted(summary.items(), key=lambda x: x[1].get('score', -9999), reverse=True)
    
    for pipeline, metrics in sorted_items:
        p_type = metrics.get('type', 'unknown')
        acc = metrics.get('avg_accuracy', 0)
        lat = metrics.get('avg_latency', 0)
        cost = metrics.get('total_cost', 0)
        fails = metrics.get('failure_count', 0)
        score = metrics.get('score', 0)
        
        print(f"{pipeline:<15} | {p_type:<15} | {acc:<10.2f} | {lat:<10.2f}s | {cost:<10.4f} | {fails:<10} | {score:<10.2f}")
    print("="*95 + "\n")
