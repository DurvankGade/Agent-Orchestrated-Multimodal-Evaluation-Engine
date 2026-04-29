from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="phi3")

def critic_node(state):
    summary = state.get("summary", {})
    analysis = {}

    for p_name, m in summary.items():
        prompt = f"""
Strictly analyze the metrics. Do NOT hallucinate.
Pipeline: {p_name}
Accuracy: {m['avg_accuracy']:.2f}
Latency: {m['avg_latency']:.2f}s
Failures: {m['failure_count']}

Format exactly:
Pipeline: {p_name}
Accuracy: {m['avg_accuracy']:.2f}
Latency: {m['avg_latency']:.2f}s
Failures: {m['failure_count']}

Strength:
Weakness:
When to use:
"""
        try:
            analysis[p_name] = llm.invoke(prompt).strip()
        except:
            analysis[p_name] = "Analysis Failed"

    state["analysis"] = analysis
    return state