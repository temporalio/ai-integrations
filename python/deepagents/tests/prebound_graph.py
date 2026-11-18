"""A fan-out graph's sync stream, bound when imported, before any Worker sets up the plugin."""

from typing import Any, TypedDict

from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)

from temporalio import workflow


class Fanout(TypedDict, total=False):
    left: bool
    right: bool


def fanout_graph(visited: list[str]) -> Any:
    def left(state: Fanout) -> Fanout:
        visited.append("left")
        return {**state, "left": workflow.in_workflow()}

    def right(state: Fanout) -> Fanout:
        visited.append("right")
        return {**state, "right": workflow.in_workflow()}

    graph = StateGraph(Fanout)
    graph.add_node("a", left)
    graph.add_node("b", right)
    for node in ("a", "b"):
        graph.add_edge(START, node)
        graph.add_edge(node, END)
    return graph.compile()


EARLY_GRAPH_VISITS: list[str] = []
EARLY_GRAPH_STREAM = fanout_graph(EARLY_GRAPH_VISITS).stream
