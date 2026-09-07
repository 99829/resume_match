from langgraph.graph import StateGraph, START, END

from src.graph.state import MainState
from src.graph.nodes import (
    parse_jd_node,
    fan_out_to_resumes,
    parse_and_match_resume_node,
    rank_node,
)


def build_graph():
    graph = StateGraph(MainState)

    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("parse_and_match_resume", parse_and_match_resume_node)
    graph.add_node("rank", rank_node)

    graph.add_edge(START, "parse_jd")
    graph.add_conditional_edges("parse_jd", fan_out_to_resumes, ["parse_and_match_resume"])
    graph.add_edge("parse_and_match_resume", "rank")
    graph.add_edge("rank", END)

    return graph.compile()


def run_bulk_match(jd_text: str, resumes: list[dict]) -> list[dict]:
    app = build_graph()
    result = app.invoke({"jd_text": jd_text, "resumes": resumes, "match_results": []})
    return result["ranked_results"]