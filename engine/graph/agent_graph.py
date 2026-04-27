def to_dict(self):
    return self.__dict__

def from_dict(self, d):
    self.__dict__.update(d)
    return self

from langgraph.graph import StateGraph, END

from engine.nodes.planner import planner_node
from engine.nodes.executor import executor_node
from engine.nodes.evaluator import evaluator_node
from engine.nodes.critic import critic_node
from engine.nodes.recommender import recommender_node

def build_graph():

    builder = StateGraph(dict)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("critic", critic_node)
    builder.add_node("recommender", recommender_node)

    builder.set_entry_point("planner")

    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "evaluator")
    builder.add_edge("evaluator", "critic")
    builder.add_edge("critic", "recommender")

    builder.add_edge("recommender", END)

    return builder.compile()