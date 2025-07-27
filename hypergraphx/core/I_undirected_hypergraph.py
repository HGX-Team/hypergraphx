from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union, Optional, Any

from hypergraphx.core.i_hypergraph import IHypergraph

class IUndirectedHypergraph(IHypergraph):
    """
    Abstract base class defining the common interface for all undirected hypergraph implementations.
    
    This class specifies the required properties and methods that must be implemented
    by Hypergraph, TemporalHypergraph, and MultiplexHypergraph.

    Extends IHypergraph with undirected-specific functionality and adjacency management.
    """

    def __init__(
        self,
        edge_list=None,
        weighted: bool = False,
        weights=None,
        hypergraph_metadata: Optional[Dict] = None,
        node_metadata: Optional[Dict] = None,
        edge_metadata: Optional[Dict] = None
    ):
        """
        Initialize an Undirected Hypergraph.

        Parameters
        ----------
        edge_list : list of tuples, optional
            A list of hyperedges, where each hyperedge is represented as a tuple of nodes.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is False.
        weights : list of floats, optional
            A list of weights corresponding to each edge in `edge_list`. Required if `weighted` is True.
        hypergraph_metadata : dict, optional
            Metadata for the hypergraph. Default is an empty dictionary.
        node_metadata : dict, optional
            A dictionary of metadata for nodes, where keys are node identifiers and values are metadata dictionaries.
        edge_metadata : list of dicts, optional
            A list of metadata dictionaries corresponding to the edges in `edge_list`.

        Raises
        ------
        ValueError
            If `edge_list` and `weights` have mismatched lengths when `weighted` is True.
        """
        # Call parent constructor
        super().__init__(
            edge_list=edge_list,
            weighted=weighted,
            weights=weights,
            hypergraph_metadata=hypergraph_metadata,
            node_metadata=node_metadata,
            edge_metadata=edge_metadata
        )
        # Initialize undirected-specific data structures
        self._adj = {}

    # =============================================================================
    # Undirected-Specific Node Management
    # =============================================================================
    
    def add_node(self,
                 node: Any,
                 metadata: Optional[Dict] = None) -> None:
        """
        Add a node to the undirected hypergraph. If the node is already in the hypergraph, nothing happens.

        Parameters
        ----------
        node : object
            The node to add.
        metadata : dict, optional
            Metadata for the node.

        Returns
        -------
        None
        """
        # Call parent method for metadata management
        super().add_node(node, metadata)
        
        # Initialize adjacency list for undirected hypergraph
        if node not in self._adj:
            self._adj[node] = []

    # =============================================================================
    # Undirected-Specific Adjacency and Structure Access
    # =============================================================================
    
    def check_node(self, node):
        """Checks if the specified node is in the hypergraph.

        Parameters
        ----------
        node : Object
            The node to check.

        Returns
        -------
        bool
            True if the node is in the hypergraph, False otherwise.

        """
        return node in self._adj
    
    def get_adj_dict(self):
        """Get the adjacency dictionary."""
        return self._adj

    def set_adj_dict(self, adj_dict):
        """Set the adjacency dictionary."""
        self._adj = adj_dict

    def get_incident_edges(self, node, order: int = None, size: int = None):
        """
        Get the incident hyperedges of a node in the hypergraph.

        Parameters
        ----------
        node : object
            The node of interest.
        order : int
            The order of the hyperedges to consider.
        size : int
            The size of the hyperedges to consider.

        Returns
        -------
        list
            The incident hyperedges of the node.

        Raises
        ------
        ValueError
            If the node is not in the hypergraph.

        """
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return list(
                [self._reverse_edge_list[edge_id] for edge_id in self._adj[node]]
            )
        else:
            if order is None:
                order = size - 1
            return list(
                [
                    self._reverse_edge_list[edge_id]
                    for edge_id in self._adj[node]
                    if len(self._reverse_edge_list[edge_id]) - 1 == order
                ]
            )