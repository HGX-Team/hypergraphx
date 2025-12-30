from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hypergraphx.core.undirected import Hypergraph

from hypergraphx.exceptions import MissingNodeError


def _bfs(hg: Hypergraph, start, max_depth=None, order=None, size=None):
    """
    Breadth-first search of the hypergraph starting from the given node.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to search.
    start : Node. The node to start the search from.
    max_depth : int. The maximum depth to search. If None, the search is not limited.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    set. The nodes visited during the search.

    Raises
    ------
    ValueError. If the start node is not in the hypergraph.
    """
    if not hg.check_node(start):
        raise MissingNodeError(f"Node {start} not in hypergraph.")
    visited = set()
    queue = deque([(start, 0)])

    add_visited = visited.add
    get_neighbors = hg.get_neighbors

    while queue:
        node, depth = queue.popleft()
        if node not in visited:
            add_visited(node)
            if max_depth is None or depth < max_depth:
                neighbors = get_neighbors(node, order=order, size=size)
                queue.extend((n, depth + 1) for n in neighbors if n not in visited)

    return visited


def _dfs(hg: Hypergraph, start, max_depth=None, order=None, size=None):
    """
    Depth-first search of the hypergraph starting from the given node.
    Parameters
    ----------
    hg : Hypergraph. The hypergraph to search.
    start : Node. The node to start the search from.
    max_depth : int. The maximum depth to search. If None, the search is not limited.
    order : int. The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int. The size of the hyperedges to consider. If None, all hyperedges are considered.

    Returns
    -------
    set. The nodes visited during the search.

    Raises
    ------
    ValueError. If the start node is not in the hypergraph.
    """
    if not hg.check_node(start):
        raise MissingNodeError(f"Node {start} not in hypergraph.")
    visited = set()
    stack = [(start, 0)]

    add_visited = visited.add
    get_neighbors = hg.get_neighbors

    while stack:
        node, depth = stack.pop()
        if node not in visited:
            add_visited(node)
            if max_depth is None or depth < max_depth:
                new_depth = depth + 1
                neighbors = get_neighbors(node, order=order, size=size)
                stack.extend((n, new_depth) for n in neighbors if n not in visited)

    return visited
