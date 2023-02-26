from hoinetx.core import Hypergraph
from hoinetx.utils.visits import bfs


def connected_components(hg: Hypergraph):
    visited = []
    components = []
    for node in hg.get_nodes():
        if node not in visited:
            component = bfs(hg, node)
            visited += component
            components.append(component)
    return components


def node_connected_component(hg: Hypergraph, node):
    return bfs(hg, node)


def num_connected_components(hg: Hypergraph):
    return len(hg.connected_components())


def largest_component(hg: Hypergraph):
    components = hg.connected_components()
    return max(components, key=len)


def largest_component_size(hg: Hypergraph):
    return len(hg.largest_component())


def isolated_nodes(hg: Hypergraph):
    return [node for node in hg.get_nodes() if len(hg.neighbors(node)) == 0]

