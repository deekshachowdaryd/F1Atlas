"""
graph_utils.py
--------------
Knowledge-graph helpers: loading the pickled networkx graph and reading
facts out of it (neighbours, human-readable descriptions).
"""
from typing import List
import pickle
import networkx as nx

from config import PICKLE_PATH


def load_graph() -> nx.DiGraph:
    if not PICKLE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found at '{PICKLE_PATH}'. Run ingest.py first."
        )
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)


def get_related_entity_ids(G: nx.DiGraph, entity_id: str, max_hops: int = 1) -> List[str]:
    if entity_id not in G:
        return []
    visited = {entity_id}
    frontier = {entity_id}
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            neighbours = set(G.successors(node)) | set(G.predecessors(node))
            next_frontier.update(neighbours - visited)
        visited.update(next_frontier)
        frontier = next_frontier
    visited.discard(entity_id)
    return list(visited)


def describe_entity(G: nx.DiGraph, entity_id: str) -> str:
    if entity_id not in G:
        return ""
    node = G.nodes[entity_id]
    lines = [f"{node.get('label', entity_id)} ({node.get('type', 'entity')})"]
    for _, target, data in G.out_edges(entity_id, data=True):
        rel = data.get("relation", "")
        t_node = G.nodes.get(target, {})
        t_lbl = t_node.get("label", target)
        if rel == "DROVE_FOR":
            lines.append(f"  Drove for {t_lbl} ({data.get('year_start')}-{data.get('year_end')})")
        elif rel == "WON_CHAMPIONSHIP":
            lines.append(f"  Won championship: {t_lbl}")
        elif rel == "RIVAL_OF":
            lines.append(f"  Rival: {t_lbl} ({data.get('label', '')})")
        elif rel == "TEAMMATE_OF":
            lines.append(f"  Teammate: {t_lbl}")
    return "\n".join(lines)
