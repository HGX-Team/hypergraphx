import json
import os

from hypergraphx import Hypergraph, TemporalHypergraph, MultiplexHypergraph


def load_hif(path: str) -> Hypergraph:
    """
    Load a hypergraph from a HIF file.

    Parameters
    ----------
    path : str
        The path to the HIF file

    Returns
    -------
    Hypergraph
        The loaded hypergraph
    """

    with open(path) as file:
        data = json.loads(file.read())

    if 'type' not in data:
        raise ValueError("Absent hypergraph type")

    if 'metadata' not in data:
        raise ValueError("Metadata is required")

    if 'nodes' not in data:
        raise ValueError("Information about nodes in the hypergraph is required")

    if 'edges' not in data:
        raise ValueError("Information about edges in the hypergraph is required")

    if 'incidences' not in data:
        raise ValueError("Information about hypergraph incidences is required")

    if data['type'] == 'undirected':
        H = Hypergraph()
    elif data['type'] == 'directed':
        H = Hypergraph(directed=True)
    elif data['type'] == 'temporal':
        H = TemporalHypergraph()
    elif data['type'] == 'multiplex':
        H = MultiplexHypergraph()
    else:
        raise ValueError("Invalid hypergraph type")

    H.add_hypergraph_metadata(data['metadata'])

    for node in data['nodes']:
        H.add_node(node['uid'])
        H.set_meta(node['uid'], node['attrs'])

    for edge in data['edges']:
        H.add_empty_edge(edge['uid'])
        H.set_meta(edge['uid'], edge['attrs'])

    tmp_edges = {}
    for incidence in data['incidences']:
        edge = incidence['edge']
        if edge not in tmp_edges:
            tmp_edges[edge] = []
        tmp_edges[edge].append(incidence['node'])

    for edge, nodes in tmp_edges.items():
        H.add_edge(nodes, uid=edge)

    return H


def save_hif(H: Hypergraph, path: str):
    """
    Save a hypergraph to a HIF file.

    Parameters
    ----------
    H: Hypergraph
        The hypergraph to save.
    path: str
        The path to save the hypergraph to.
    """
    data = {}

    data['metadata'] = H.get_hypergraph_metadata()
    if isinstance(H, TemporalHypergraph):
        data['type'] = 'temporal'
    elif isinstance(H, MultiplexHypergraph):
        data['type'] = 'multiplex'
    elif H.is_directed():
        data['type'] = 'directed'
    else:
        data['type'] = 'undirected'

    data['nodes'] = []
    for node in H.get_nodes():
        data['nodes'].append({'uid': node, 'attrs': H.get_meta(node)})

    data['edges'] = []
    for edge in H.get_edges():
        data['edges'].append({'uid': edge, 'attrs': H.get_meta(edge)})

    data['incidences'] = []
    for edge in H.get_edges():
        uid = H.get_edge_uid(edge)
        for node in edge:
            data['incidences'].append({'edge': uid, 'node': node})

    with open(path, 'w') as file:
        file.write(json.dumps(data))



