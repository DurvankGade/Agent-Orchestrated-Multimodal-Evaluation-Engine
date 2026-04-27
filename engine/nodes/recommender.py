def recommender_node(state):

    best = max(state["results"], key=lambda x: x["score"])
    state["best"] = best

    state["decision_reason"] = f"""
    Selected {best['pipeline']} because it achieved the highest score
    by balancing accuracy ({best['accuracy']}) and latency ({best['latency']:.2f}s).
    """

    print(f"[Recommender] Best: {best['pipeline']}")
    return state