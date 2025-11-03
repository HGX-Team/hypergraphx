import copy
import warnings
from abc import abstractmethod
from typing import Tuple, Any, List, Dict, Optional, Set, Union
from collections import Counter

from hypergraphx.core.i_hypergraph import IHypergraph


class IUndirectedHypergraph(IHypergraph):
    """
    Abstract base class for undirected hypergraphs that provides common functionality
    for adjacency list management, canonical edge handling, and shared operations.
    
    This class serves as an intermediate layer between IHypergraph and concrete
    implementations like Hypergraph, MultiplexHypergraph, and TemporalHypergraph.
    """

    def __init__(
        self,
        edge_list: Optional[List] = None,
        weighted: bool = False,
        weights: Optional[List[int]] = None,
        hypergraph_metadata: Optional[Dict] = None,
        node_metadata: Optional[Dict] = None,
        edge_metadata: Optional[List[Dict]] = None
    ):
        """
        Initialize the undirected hypergraph base class.

        Parameters
        ----------
        edge_list : list of tuples, optional
            A list of hyperedges, where each hyperedge is represented as a tuple of nodes.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is False.
        weights : list of floats, optional
            A list of weights corresponding to each edge in `edge_list`.
        hypergraph_metadata : dict, optional
            Metadata for the hypergraph. Default is an empty dictionary.
        node_metadata : dict, optional
            A dictionary of metadata for nodes.
        edge_metadata : list of dicts, optional
            A list of metadata dictionaries corresponding to the edges.
        """
        super().__init__(
            edge_list=edge_list,
            weighted=weighted,
            weights=weights,
            hypergraph_metadata=hypergraph_metadata,
            node_metadata=node_metadata,
            edge_metadata=edge_metadata
        )
        
        # Initialize adjacency list - common to all undirected hypergraph implementations
        self._adj = {}

    # =============================================================================
    # Common Node Management Implementation
    # =============================================================================

    def add_node(self, node: Any, metadata: Optional[Dict] = None) -> None:
        """
        Add a node to the hypergraph. If the node is already in the hypergraph, nothing happens.

        Parameters
        ----------
        node : object
            The node to add.
        metadata : dict, optional
            Metadata for the node.
        """
        # Call parent implementation for metadata handling
        super().add_node(node, metadata)
        
        # Add to adjacency list if not already present
        if node not in self._adj:
            self._adj[node] = []

    def get_nodes(self, metadata: bool = False):
        """
        Get all nodes in the hypergraph.

        Parameters
        ----------
        metadata : bool, optional
            If True, return node metadata dictionary. If False, return list of nodes.

        Returns
        -------
        list or dict
            List of nodes or dictionary of node metadata.
        """
        if metadata:
            return {node: self.get_node_metadata(node) for node in self._adj.keys()}
        else:
            return list(self._adj.keys())

    def check_node(self, node: Any) -> bool:
        """
        Check if a node exists in the hypergraph.

        Parameters
        ----------
        node : object
            The node to check.

        Returns
        -------
        bool
            True if the node exists, False otherwise.
        """
        return node in self._adj

    def remove_nodes(self, node_list: List[Any], keep_edges: bool = False) -> None:
        """
        Remove a list of nodes from the hypergraph.

        Parameters
        ----------
        node_list : list
            The list of nodes to remove.
        keep_edges : bool, optional
            If True, edges incident to the nodes are kept but updated to exclude the nodes.
            If False, edges incident to the nodes are removed entirely. Default is False.
        """
        for node in node_list:
            self.remove_node(node, keep_edges=keep_edges)

    # =============================================================================
    # Common Edge Management Implementation
    # =============================================================================

    def add_edges(self, edge_list: List, weights: Optional[List] = None, metadata: Optional[List[Dict]] = None, **kwargs) -> None:
        """
        Add multiple edges to the hypergraph.

        Parameters
        ----------
        edge_list : list
            The list of edges to add.
        weights : list, optional
            The list of weights for the edges.
        metadata : list, optional
            The list of metadata dictionaries for the edges.
        **kwargs
            Additional parameters specific to subclass implementations.
        """
        if weights is not None and not self._weighted:
            warnings.warn(
                "Weights are provided but the hypergraph is not weighted. The weights will be ignored.",
                UserWarning,
            )
            self._weighted = True

        if self._weighted and weights is not None:
            if len(set(edge_list)) != len(list(edge_list)):
                raise ValueError(
                    "If weights are provided, the edge list must not contain repeated edges."
                )
            if len(list(edge_list)) != len(list(weights)):
                raise ValueError("The number of edges and weights must be the same.")

        for i, edge in enumerate(edge_list):
            weight = weights[i] if self._weighted and weights is not None else None
            edge_metadata = metadata[i] if metadata is not None else None
            self._add_edge_implementation(edge, weight, edge_metadata, **kwargs)

    @abstractmethod
    def _add_edge_implementation(self, edge, weight, metadata, **kwargs):
        """
        Abstract method for adding a single edge. Must be implemented by subclasses.
        
        Parameters
        ----------
        edge : tuple
            The edge to add.
        weight : float, optional
            The weight of the edge.
        metadata : dict, optional
            The metadata of the edge.
        **kwargs
            Additional parameters specific to subclass implementations.
        """
        pass

    def remove_edges(self, edge_list: List) -> None:
        """
        Remove multiple edges from the hypergraph.

        Parameters
        ----------
        edge_list : list
            The list of edges to remove.
        """
        for edge in edge_list:
            self.remove_edge(edge)

    # =============================================================================
    # Common Neighbor and Incident Edge Methods
    # =============================================================================

    def get_neighbors(self, node: Any, order: int = None, size: int = None) -> Set:
        """
        Get the neighbors of a node in the hypergraph.

        Parameters
        ----------
        node : object
            The node of interest.
        order : int, optional
            The order of the hyperedges to consider.
        size : int, optional
            The size of the hyperedges to consider.

        Returns
        -------
        set
            The neighbors of the node.

        Raises
        ------
        ValueError
            If the node is not in the hypergraph or if both order and size are specified.
        """
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
            
        neigh = set()
        edges = self.get_incident_edges(node, order=order, size=size)
        for edge in edges:
            # Extract nodes from edge (handling different edge formats)
            edge_nodes = self._extract_nodes_from_edge(edge)
            neigh.update(edge_nodes)
        
        # Remove the node itself from its neighbors
        neigh.discard(node)
        return neigh

    @abstractmethod
    def _extract_nodes_from_edge(self, edge) -> List:
        """
        Extract node list from an edge representation.
        Must be implemented by subclasses based on their edge format.
        
        Parameters
        ----------
        edge : object
            The edge representation.
            
        Returns
        -------
        list
            List of nodes in the edge.
        """
        pass

    # =============================================================================
    # Common Structural Information Methods
    # =============================================================================

    def num_edges(self, order: int = None, size: int = None, up_to: bool = False) -> int:
        """
        Get the number of edges in the hypergraph.

        Parameters
        ----------
        order : int, optional
            The order of edges to count.
        size : int, optional
            The size of edges to count.
        up_to : bool, optional
            If True, count edges up to the specified order/size.

        Returns
        -------
        int
            Number of edges matching the criteria.
        """
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")

        if order is None and size is None:
            return len(self._edge_list)
        else:
            if size is not None:
                order = size - 1
            count = 0
            for edge_key in self._edge_list:
                edge_size = self._get_edge_size(edge_key)
                edge_order = edge_size - 1
                if not up_to:
                    if edge_order == order:
                        count += 1
                else:
                    if edge_order <= order:
                        count += 1
            return count

    @abstractmethod
    def _get_edge_size(self, edge_key) -> int:
        """
        Get the size of an edge given its key representation.
        Must be implemented by subclasses based on their edge format.
        
        Parameters
        ----------
        edge_key : object
            The edge key representation.
            
        Returns
        -------
        int
            Size of the edge.
        """
        pass

    def get_sizes(self) -> List[int]:
        """
        Get the sizes of all edges in the hypergraph.

        Returns
        -------
        list
            List of edge sizes.
        """
        return [self._get_edge_size(edge_key) for edge_key in self._edge_list.keys()]

    def is_uniform(self) -> bool:
        """
        Check if the hypergraph is uniform (all edges have the same size).

        Returns
        -------
        bool
            True if the hypergraph is uniform, False otherwise.
        """
        if not self._edge_list:
            return True
            
        sizes = self.get_sizes()
        return len(set(sizes)) <= 1

    # =============================================================================
    # Common Weight Management Methods
    # =============================================================================

    def get_weights(self, order: int = None, size: int = None, up_to: bool = False, asdict: bool = False):
        """
        Get weights of edges in the hypergraph.

        Parameters
        ----------
        order : int, optional
            The order of edges to get weights for.
        size : int, optional
            The size of edges to get weights for.
        up_to : bool, optional
            If True, get weights for edges up to the specified order/size.
        asdict : bool, optional
            If True, return as dictionary mapping edges to weights.

        Returns
        -------
        list or dict
            Weights of the edges.
        """
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
            
        if order is None and size is None:
            w = {
                edge: self._weights[self._edge_list[edge]] for edge in self._edge_list.keys()
            }
        else:
            if size is not None:
                order = size - 1
            w = {}
            for edge_key in self._edge_list:
                edge_size = self._get_edge_size(edge_key)
                edge_order = edge_size - 1
                if not up_to:
                    if edge_order == order:
                        w[edge_key] = self._weights[self._edge_list[edge_key]]
                else:
                    if edge_order <= order:
                        w[edge_key] = self._weights[self._edge_list[edge_key]]

        return w if asdict else list(w.values())

    # =============================================================================
    # Common Utility Methods
    # =============================================================================

    def _canon_edge(self, edge: Tuple) -> Tuple:
        """
        Get the canonical form of an edge by sorting its components.
        This default implementation works for simple undirected edges.
        Subclasses can override for more complex edge structures.

        Parameters
        ----------
        edge : tuple
            The edge to canonicalize.

        Returns
        -------
        tuple
            The canonical form of the edge.
        """
        return tuple(sorted(edge))

    def get_adj_dict(self) -> Dict:
        """
        Get the adjacency dictionary.

        Returns
        -------
        dict
            The adjacency dictionary mapping nodes to lists of incident edge IDs.
        """
        return self._adj

    def set_adj_dict(self, adj: Dict) -> None:
        """
        Set the adjacency dictionary.

        Parameters
        ----------
        adj : dict
            The adjacency dictionary to set.
        """
        self._adj = adj

    def clear(self) -> None:
        """Clear all data from the hypergraph."""
        super().clear()
        self._adj.clear()

    # =============================================================================
    # Common Analysis Methods (delegated to external modules)
    # =============================================================================

    def degree(self, node: Any, order: int = None, size: int = None):
        """Get the degree of a node."""
        from hypergraphx.measures.degree import degree
        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order: int = None, size: int = None):
        """Get the degree sequence of the hypergraph."""
        from hypergraphx.measures.degree import degree_sequence
        return degree_sequence(self, order=order, size=size)

    def degree_distribution(self, order: int = None, size: int = None):
        """Get the degree distribution of the hypergraph."""
        from hypergraphx.measures.degree import degree_distribution
        return degree_distribution(self, order=order, size=size)

    def isolated_nodes(self, size: int = None, order: int = None):
        """Get isolated nodes in the hypergraph."""
        from hypergraphx.utils.cc import isolated_nodes
        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node: Any, size: int = None, order: int = None):
        """Check if a node is isolated."""
        from hypergraphx.utils.cc import is_isolated
        return is_isolated(self, node, size=size, order=order)

    # Connected Components
    def is_connected(self, size: int = None, order: int = None):
        """Check if the hypergraph is connected."""
        from hypergraphx.utils.cc import is_connected
        return is_connected(self, size=size, order=order)

    def connected_components(self, size: int = None, order: int = None):
        """Get connected components of the hypergraph."""
        from hypergraphx.utils.cc import connected_components
        return connected_components(self, size=size, order=order)

    def node_connected_component(self, node: Any, size: int = None, order: int = None):
        """Get the connected component containing a specific node."""
        from hypergraphx.utils.cc import node_connected_component
        return node_connected_component(self, node, size=size, order=order)

    def num_connected_components(self, size: int = None, order: int = None):
        """Get the number of connected components."""
        from hypergraphx.utils.cc import num_connected_components
        return num_connected_components(self, size=size, order=order)

    def largest_component(self, size: int = None, order: int = None):
        """Get the largest connected component."""
        from hypergraphx.utils.cc import largest_component
        return largest_component(self, size=size, order=order)

    def largest_component_size(self, size: int = None, order: int = None):
        """Get the size of the largest connected component."""
        from hypergraphx.utils.cc import largest_component_size
        return largest_component_size(self, size=size, order=order)

    # Matrix operations
    def binary_incidence_matrix(self, return_mapping: bool = False):
        """Get the binary incidence matrix."""
        from hypergraphx.linalg import binary_incidence_matrix
        return binary_incidence_matrix(self, return_mapping)

    def incidence_matrix(self, return_mapping: bool = False):
        """Get the incidence matrix."""
        from hypergraphx.linalg import incidence_matrix
        return incidence_matrix(self, return_mapping)

    def adjacency_matrix(self, return_mapping: bool = False):
        """Get the adjacency matrix."""
        from hypergraphx.linalg import adjacency_matrix
        return adjacency_matrix(self, return_mapping)

    def dual_random_walk_adjacency(self, return_mapping: bool = False):
        """Get the dual random walk adjacency matrix."""
        from hypergraphx.linalg import dual_random_walk_adjacency
        return dual_random_walk_adjacency(self, return_mapping)

    def adjacency_factor(self, t: int = 0):
        """Get the adjacency factor."""
        from hypergraphx.linalg import adjacency_factor
        return adjacency_factor(self, t)

    # =============================================================================
    # Common Serialization Support
    # =============================================================================

    def expose_data_structures(self) -> Dict:
        """
        Expose the internal data structures for serialization.
        Base implementation that subclasses can extend.

        Returns
        -------
        dict
            A dictionary containing the internal data structures.
        """
        base_data = super().expose_data_structures()
        base_data["_adj"] = self._adj
        return base_data

    def populate_from_dict(self, data: Dict) -> None:
        """
        Populate the attributes from a dictionary.
        Base implementation that subclasses can extend.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate.
        """
        super().populate_from_dict(data)
        self._adj = data.get("_adj", {})

    # =============================================================================
    # Abstract Methods That Must Be Implemented by Subclasses
    # =============================================================================

    @abstractmethod
    def remove_node(self, node: Any, keep_edges: bool = False) -> None:
        """
        Remove a node from the hypergraph.
        Must be implemented by subclasses due to edge format differences.
        """
        pass

    @abstractmethod
    def add_edge(self, edge, weight=None, metadata=None, **kwargs) -> None:
        """
        Add an edge to the hypergraph.
        Must be implemented by subclasses due to edge format differences.
        """
        pass

    @abstractmethod
    def remove_edge(self, edge, **kwargs) -> None:
        """
        Remove an edge from the hypergraph.
        Must be implemented by subclasses due to edge format differences.
        """
        pass

    @abstractmethod
    def get_edges(self, metadata: bool = False, **kwargs):
        """
        Get edges from the hypergraph.
        Must be implemented by subclasses due to edge format differences.
        """
        pass

    @abstractmethod
    def get_incident_edges(self, node: Any, order: int = None, size: int = None):
        """
        Get incident edges of a node.
        Must be implemented by subclasses due to edge format differences.
        """
        pass