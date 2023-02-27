from hoinetx.core import Hypergraph


def _bfs(hg: Hypergraph, start, max_depth=None, order=None, size=None):
    """Breadth-first search of the hypergraph starting from the given node.
    """
    visited = set()
    queue = [(start, 0)]
    while queue:
        node, depth = queue.pop(0)
        if node not in visited:
            visited.add(node)
            if max_depth is None or depth < max_depth:
                queue.extend((n, depth + 1) for n in hg.get_neighbors(node, order=order, size=size))
    return visited


def _dfs(hg: Hypergraph, start, max_depth=None, order=None, size=None):
    """Depth-first search of the hypergraph starting from the given node.
    """
    visited = set()
    stack = [(start, 0)]
    while stack:
        node, depth = stack.pop()
        if node not in visited:
            visited.add(node)
            if max_depth is None or depth < max_depth:
                stack.extend((n, depth + 1) for n in hg.get_neighbors(node, order=order, size=size))
    return visited
