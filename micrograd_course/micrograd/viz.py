"""计算图遍历；不依赖 Graphviz。"""

from __future__ import annotations

from .engine import Value


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    nodes = {root}
    edges = set()
    for node in root._prev:
        edges.add((node, root))
        next_nodes, next_edges = trace(node)
        nodes |= next_nodes
        edges |= next_edges

    return (nodes, edges)
