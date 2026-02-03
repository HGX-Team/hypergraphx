from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class LabelEncoder:
    """
    Minimal drop-in replacement for sklearn's LabelEncoder, used throughout HypergraphX
    to map arbitrary node labels to dense integer ids.

    This exists to avoid forcing scikit-learn as a hard dependency.
    """

    classes_: List[Any] | None = None
    _to_int: dict[Any, int] | None = None

    def fit(self, y: Sequence[Any]) -> "LabelEncoder":
        # Keep first-seen order to preserve a stable mapping based on the input.
        classes: List[Any] = []
        to_int: dict[Any, int] = {}
        for item in y:
            if item in to_int:
                continue
            to_int[item] = len(classes)
            classes.append(item)
        self.classes_ = classes
        self._to_int = to_int
        return self

    def transform(self, y: Iterable[Any]) -> np.ndarray:
        if self._to_int is None:
            raise ValueError("LabelEncoder is not fitted. Call fit() first.")
        return np.array([self._to_int[item] for item in y], dtype=int)

    def inverse_transform(self, y: Iterable[int]) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("LabelEncoder is not fitted. Call fit() first.")
        classes = self.classes_
        return np.array([classes[int(i)] for i in y], dtype=object)


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
    if mapping.classes_ is None:
        raise ValueError("LabelEncoder is not fitted. Call fit() first.")
    return {i: node for i, node in enumerate(mapping.classes_)}


def relabel_edges_with_mapping(edges: List[Tuple], mapping: dict):
    """
    Relabel edges using a dictionary mapping old labels to new labels.

    Parameters
    ----------
    edges: List[Tuple]
        The edges to relabel
    mapping: dict
        Mapping from old labels to new labels

    Returns
    -------
    List[Tuple]
        The relabeled edges
    """
    res = []
    for edge in edges:
        new_edge = [mapping[v] for v in edge]
        res.append(tuple(sorted(new_edge)))
    return sorted(res)
