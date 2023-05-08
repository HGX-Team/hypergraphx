from typing import List, Tuple

from sklearn.preprocessing import LabelEncoder


def relabel_edge(mapping: LabelEncoder, edge: Tuple):
    """
    Relabel an edge using a mapping.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to use
    edge: Tuple
        The edge to relabel

    Returns
    -------
    Tuple
        The relabeled edge
    """
    return tuple(mapping.transform(edge))


def relabel_edges(mapping: LabelEncoder, edges: List[Tuple]):
    """
    Relabel a list of edges using a mapping.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to use
    edges: List[Tuple]
        The edges to relabel

    Returns
    -------
    List[Tuple]
        The relabeled edges
    """
    return [relabel_edge(mapping, edge) for edge in edges]


def inverse_relabel_edge(mapping: LabelEncoder, edge: Tuple):
    """
    Revert edge relabeling using a mapping.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to use
    edge: Tuple
        The edge to relabel

    Returns
    -------
    Tuple
        The relabeled edge
    """
    return tuple(mapping.inverse_transform(edge))


def inverse_relabel_edges(mapping: LabelEncoder, edges: List[Tuple]):
    """
    Revert edges relabeling using a mapping.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to use
    edges: List[Tuple]
        The edges to relabel

    Returns
    -------
    List[Tuple]
        The relabeled edges
    """
    return [inverse_relabel_edge(mapping, edge) for edge in edges]


def map_node(mapping: LabelEncoder, node):
    """
    Map a node using a mapping.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to use
    node: Any
        The node to map

    Returns
    -------
    Any
        The mapped node
    """
    return mapping.transform([node])[0]


def map_nodes(mapping: LabelEncoder, nodes: List):
    """
    Map a list of nodes using a mapping.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to use
    nodes: List
        The nodes to map

    Returns
    -------
    List
        The mapped nodes
    """
    return mapping.transform(nodes)


def inverse_map_nodes(mapping: LabelEncoder, nodes: List):
    """
    Revert node mapping using a mapping.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to use
    nodes: List
        The nodes to map

    Returns
    -------
    List
        The mapped nodes
    """
    return mapping.inverse_transform(nodes)


def get_inverse_mapping(mapping: LabelEncoder):
    """
    Get the inverse mapping of a LabelEncoder.

    Parameters
    ----------
    mapping: LabelEncoder
        The mapping to invert

    Returns
    -------
    dict
        The inverse mapping
    """
    return dict(zip(mapping.transform(mapping.classes_), mapping.classes_))