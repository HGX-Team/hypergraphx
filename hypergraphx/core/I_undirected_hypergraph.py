from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union, Optional, Any

from hypergraphx.core.I_hypergraph import IHypergraph

def _canon_edge(edge):
    """Canonicalize an edge by sorting its components."""
    edge = tuple(edge)

    if len(edge) == 2:
        if isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
            # Sort the inner tuples and return
            return (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        elif not isinstance(edge[0], tuple) and not isinstance(edge[1], tuple):
            # Sort the edge itself if it contains IDs (non-tuple elements)
            return tuple(sorted(edge))

    return tuple(sorted(edge))


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
        
        # Add nodes from node_metadata to adjacency structure
        if node_metadata:
            for node in node_metadata.keys():
                if node not in self._adj:
                    self._adj[node] = []

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
    
    def get_adj_dict(self):
        """Get the adjacency dictionary."""
        return self._adj

    def set_adj_dict(self, adj_dict):
        """Set the adjacency dictionary."""
        self._adj = adj_dict

    # =============================================================================
    # Abstract Methods (Must be implemented by subclasses)
    # =============================================================================
    
    @abstractmethod
    def add_edge(self, edge, *args, **kwargs) -> None:
        """
        Add an edge to the hypergraph.
        
        Note: Signature varies by implementation:
        - Hypergraph: add_edge(edge, weight=None, metadata=None)
        - MultiplexHypergraph: add_edge(edge, layer, weight=None, metadata=None)
        - TemporalHypergraph: add_edge(edge, time, weight=None, metadata=None)
        """
        pass

    @abstractmethod
    def add_edges(self, edge_list, *args, **kwargs) -> None:
        """
        Add multiple edges to the hypergraph.
        
        Note: Signature varies by implementation:
        - Hypergraph: add_edges(edge_list, weights=None, metadata=None)
        - MultiplexHypergraph: add_edges(edge_list, edge_layer, weights=None, metadata=None)
        - TemporalHypergraph: add_edges(edge_list, time_list, weights=None, metadata=None)
        """
        pass

    @abstractmethod
    def remove_edge(self, edge, *args, **kwargs) -> None:
        """
        Remove an edge from the hypergraph.
        
        Note: Signature varies by implementation:
        - Hypergraph: remove_edge(edge)
        - MultiplexHypergraph: remove_edge(edge) where edge is ((nodes...), layer)
        - TemporalHypergraph: remove_edge(edge, time)
        """
        pass

    @abstractmethod
    def get_edges(self, *args, **kwargs):
        """
        Get edges from the hypergraph.
        
        Note: Parameters vary by implementation due to different filtering capabilities.
        """
        pass

    @abstractmethod
    def check_edge(self, edge, *args, **kwargs) -> bool:
        """
        Check if an edge exists in the hypergraph.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def get_weight(self, edge, *args, **kwargs):
        """
        Get the weight of an edge.
        
        Note: Signature varies by implementation:
        - Hypergraph: get_weight(edge)
        - MultiplexHypergraph: get_weight(edge, layer)
        - TemporalHypergraph: get_weight(edge, time)
        """
        pass

    @abstractmethod
    def set_weight(self, edge, *args, **kwargs) -> None:
        """
        Set the weight of an edge.
        
        Note: Signature varies by implementation:
        - Hypergraph: set_weight(edge, weight)
        - MultiplexHypergraph: set_weight(edge, layer, weight)
        - TemporalHypergraph: set_weight(edge, time, weight)
        """
        pass

    @abstractmethod
    def get_weights(self, *args, **kwargs):
        """
        Get weights of edges in the hypergraph.
        
        Note: Parameters vary by implementation.
        """
        pass

    @abstractmethod
    def get_sizes(self) -> List[int]:
        """
        Get the size of each edge in the hypergraph.

        Returns
        -------
        list
            A list of integers representing the size of each edge.
        """
        pass

    @abstractmethod
    def get_orders(self) -> List[int]:
        """
        Get the order of each edge in the hypergraph.

        Returns
        -------
        list
            A list of integers representing the order of each edge.
        """
        pass

    # =============================================================================
    # Abstract Metadata Methods (Implementation-specific signatures)
    # =============================================================================
    
    @abstractmethod
    def get_edge_metadata(self, edge, *args, **kwargs):
        """
        Get metadata for a specific edge.
        
        Note: Signature varies by implementation:
        - Hypergraph: get_edge_metadata(edge)
        - MultiplexHypergraph: get_edge_metadata(edge, layer)
        - TemporalHypergraph: get_edge_metadata(edge, time)
        """
        pass

    @abstractmethod
    def set_edge_metadata(self, edge, *args, **kwargs):
        """
        Set metadata for a specific edge.
        
        Note: Signature varies by implementation:
        - Hypergraph: set_edge_metadata(edge, metadata)
        - MultiplexHypergraph: set_edge_metadata(edge, layer, metadata)
        - TemporalHypergraph: set_edge_metadata(edge, time, metadata)
        """
        pass

    @abstractmethod
    def set_attr_to_edge_metadata(self, edge, *args, field, value):
        """
        Set an attribute in edge metadata.
        
        Note: Signature varies by implementation:
        - Hypergraph: set_attr_to_edge_metadata(edge, field, value)
        - MultiplexHypergraph: set_attr_to_edge_metadata(edge, layer, field, value)
        - TemporalHypergraph: set_attr_to_edge_metadata(edge, time, field, value)
        """
        pass

    @abstractmethod
    def remove_attr_from_edge_metadata(self, edge, *args, field):
        """
        Remove an attribute from edge metadata.
        
        Note: Signature varies by implementation:
        - Hypergraph: remove_attr_from_edge_metadata(edge, field)
        - MultiplexHypergraph: remove_attr_from_edge_metadata(edge, layer, field)
        - TemporalHypergraph: remove_attr_from_edge_metadata(edge, time, field)
        """
        pass

    # =============================================================================
    # Serialization Support (Abstract - implementation-specific)
    # =============================================================================
    
    @abstractmethod
    def expose_data_structures(self) -> Dict:
        """
        Expose the internal data structures of the hypergraph for serialization.

        Returns
        -------
        dict
            A dictionary containing all internal attributes of the hypergraph.
        """
        pass

    @abstractmethod
    def populate_from_dict(self, data: Dict) -> None:
        """
        Populate the attributes of the hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        pass

    @abstractmethod
    def expose_attributes_for_hashing(self) -> Dict:
        """
        Expose relevant attributes for hashing.

        Returns
        -------
        dict
            A dictionary containing key attributes for hashing.
        """
        pass