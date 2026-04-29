from langgraph.graph import StateGraph, END
from engine.nodes.planner import planner_node
from engine.nodes.executor import executor_node
from engine.nodes.evaluator import evaluator_node
from engine.nodes.aggregator import aggregator_node
from engine.nodes.critic import critic_node
from engine.nodes.recommender import recommender_node

def build_graph():
    g = StateGraph(dict)

    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("aggregator", aggregator_node)
    g.add_node("critic", critic_node)
    g.add_node("recommender", recommender_node)

    g.set_entry_point("planner")

    g.add_edge("planner", "executor")
    g.add_edge("executor", "evaluator")
    g.add_edge("evaluator", "aggregator")
    g.add_edge("aggregator", "critic")
    g.add_edge("critic", "recommender")
    g.add_edge("recommender", END)

    return g.compile()