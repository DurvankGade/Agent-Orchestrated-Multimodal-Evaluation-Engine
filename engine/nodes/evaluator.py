from evaluators.composite import score

def evaluator_node(state):

    ground_truth = state.get("ground_truth", "")

    state["results"] = [
        score(r, ground_truth) for r in state["results"]
    ]
    print(f"[Evaluator] Scored {len(state['results'])} results")
    return state