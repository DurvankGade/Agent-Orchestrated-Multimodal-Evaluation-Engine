from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")

def critic_node(state):

    analysis = {}

    for r in state["results"]:
        prompt = f"""
Analyze this pipeline performance briefly.

Pipeline: {r['pipeline']}
Accuracy: {r.get('accuracy')}
Latency: {r.get('latency')}

Give:
1. Strength
2. Weakness
3. When to use

Keep it under 3 lines.
"""

        try:
            explanation = llm.invoke(prompt)
        except:
            explanation = "LLM failed"

        analysis[r["pipeline"]] = explanation.strip()

    state["analysis"] = analysis

    return state