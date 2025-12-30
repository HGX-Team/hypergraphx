import json
import logging

from hypergraphx import Hypergraph


def read_hif(path: str) -> Hypergraph:
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
    edge_name_to_uid = {}
    node_name_to_uid = {}
    eid = 0
    nid = 0

    with open(path) as file:
        data = json.loads(file.read())

    if "type" not in data:
        logging.getLogger(__name__).warning("No hypergraph type - assume undirected")
        data["type"] = "undirected"

    if data["type"] == "undirected" or data["type"] == "asc":
        H = Hypergraph()
    elif data["type"] == "directed":
        H = Hypergraph(directed=True)
    else:
        raise ValueError(f"Unknown hypergraph type: {data['type']}")

    if "metadata" in data:
        H.set_hypergraph_metadata(data["metadata"])

    tmp_edges = {}
    for incidence in data["incidences"]:
        if incidence["edge"] not in edge_name_to_uid:
            edge_name_to_uid[incidence["edge"]] = eid
            eid += 1
        edge = edge_name_to_uid[incidence["edge"]]

        if incidence["node"] not in node_name_to_uid:
            node_name_to_uid[incidence["node"]] = nid
            nid += 1
        node = node_name_to_uid[incidence["node"]]

        if edge not in tmp_edges:
            tmp_edges[edge] = []
        tmp_edges[edge].append(node)

    for record in data["nodes"]:
        node_name = record["node"]
        if node_name not in node_name_to_uid:
            node_name_to_uid[node_name] = nid
            nid += 1
        node = node_name_to_uid[node_name]
        H.add_node(node)
        H.set_node_metadata(node, record)

    added = {}

    for record in data["edges"]:
        edge_name = record["edge"]
        if edge_name not in edge_name_to_uid:
            edge_name_to_uid[edge_name] = eid
            eid += 1
        edge = edge_name_to_uid[edge_name]
        if edge in tmp_edges:
            H.add_edge(tuple(sorted(tmp_edges[edge])))
            added[tuple(sorted(tmp_edges[edge]))] = True
            H.set_edge_metadata(tuple(sorted(tmp_edges[edge])), record)
        else:
            H.add_empty_edge(edge_name, record)

    for incidence in data["incidences"]:
        edge = edge_name_to_uid[incidence["edge"]]
        node = node_name_to_uid[incidence["node"]]
        if tuple(sorted(tmp_edges[edge])) not in added:
            H.add_edge(tuple(sorted(tmp_edges[edge])))
            added[tuple(sorted(tmp_edges[edge]))] = True
        H.set_incidence_metadata(tuple(sorted(tmp_edges[edge])), node, incidence)

    return H


def write_hif(H: Hypergraph, path: str):
    """
    Save a hypergraph to a HIF file.

    Parameters
    ----------
    H: Hypergraph
        The hypergraph to save.
    path: str
        The path to save the hypergraph to.
    """

    data = {
        "type": "undirected",
        "metadata": H.get_hypergraph_metadata(),
        "edges": H.get_all_edges_metadata(),
        "nodes": H.get_all_nodes_metadata(),
        "incidences": H.get_all_incidences_metadata(),
    }

    with open(path, "w") as file:
        file.write(json.dumps(data))
