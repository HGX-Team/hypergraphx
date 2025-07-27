from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union, Optional, Any
import copy
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from hypergraphx.measures.degree import degree
from hypergraphx.utils.cc import is_isolated
from hypergraphx.utils.cc import isolated_nodes
from hypergraphx.measures.degree import degree_sequence


class IHypergraph(ABC):
    """
    Abstract base class defining the common interface for all hypergraph implementations.
    
    This class specifies the required properties and methods that must be implemented
    by all hypergraph types: Hypergraph, TemporalHypergraph, MultiplexHypergraph, and DirectedHypergraph.

    Contains functionality common to both directed and undirected hypergraphs.
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
        Initialize a Hypergraph.

        Parameters
        ----------
        edge_list : list, optional
            A list of hyperedges. Format varies by implementation.
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
        # Initialize hypergraph metadata
        self._hypergraph_metadata = hypergraph_metadata or {}
        self._hypergraph_metadata.update({"weighted": weighted})

        self._weighted = weighted
        self._weights = {}
        self._node_metadata = node_metadata or {}
        self._edge_metadata = edge_metadata or {}
        self._edge_list = {}
        self._reverse_edge_list = {}
        self._next_edge_id = 0

        self._incidences_metadata = {}

        # Add node metadata if provided
        if node_metadata:
            for node, metadata in node_metadata.items():
                self.add_node(node, metadata=metadata)

    # =============================================================================
    # Node Management (Shared Implementation)
    # =============================================================================
    
    def add_node(self,
                 node: Any,
                 metadata: Optional[Dict] = None) -> None:
        """
        Add a node to the hypergraph. If the node is already in the hypergraph, nothing happens.

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
        if metadata is None:
            metadata = {}
        # Implementation varies by subclass due to different adjacency structures
        if node not in self._node_metadata:
            self._node_metadata[node] = metadata

    def add_nodes(self,
                  node_list: List[Any],
                  metadata: Optional[Dict] = None) -> None:
        """
        Add a list of nodes to the hypergraph.

        Parameters
        ----------
        node_list : list
            The list of nodes to add.
        metadata : dict, optional
            Dictionary mapping nodes to their metadata.

        Returns
        -------
        None
        """
        for node in node_list:
            node_metadata = None
            if metadata is not None:
                if node in metadata:
                    node_metadata = metadata[node]
                else:
                    raise ValueError(
                        "The metadata dictionary must contain an entry for each node in the node list."
                    )
            self.add_node(node, metadata=node_metadata)

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
            return self._node_metadata
        else:
            return list(self._node_metadata.keys())

    @abstractmethod
    def remove_node(self, node: Any, keep_edges: bool = False) -> None:
        """
        Remove a node from the hypergraph.

        Parameters
        ----------
        node : object
            The node to remove.
        keep_edges : bool, optional
            If True, edges incident to the node are kept but updated to exclude the node.
            If False, edges incident to the node are removed entirely. Default is False.

        Raises
        ------
        ValueError
            If the node is not in the hypergraph.
        """
        pass

    @abstractmethod
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
        pass

    # =============================================================================
    # Edge Management (Abstract - varies by implementation)
    # =============================================================================
    
    @abstractmethod
    def add_edge(self, edge, *args, **kwargs) -> None:
        """
        Add an edge to the hypergraph.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def add_edges(self, edge_list, *args, **kwargs) -> None:
        """
        Add multiple edges to the hypergraph.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def remove_edge(self, edge, *args, **kwargs) -> None:
        """
        Remove an edge from the hypergraph.
        
        Note: Signature varies by implementation.
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

    def get_edge_list(self):
        """Get the edge list dictionary."""
        return self._edge_list

    def set_edge_list(self, edge_list):
        """Set the edge list dictionary."""
        self._edge_list = edge_list

    # =============================================================================
    # Weight Management (Abstract - varies by implementation)
    # =============================================================================
    
    @abstractmethod
    def get_weight(self, edge, *args, **kwargs):
        """
        Get the weight of an edge.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def set_weight(self, edge, *args, **kwargs) -> None:
        """
        Set the weight of an edge.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def get_weights(self, *args, **kwargs):
        """
        Get weights of edges in the hypergraph.
        
        Note: Parameters vary by implementation.
        """
        pass

    def is_weighted(self) -> bool:
        """
        Check if the hypergraph is weighted.

        Returns
        -------
        bool
            True if the hypergraph is weighted, False otherwise.
        """
        return self._weighted

    # =============================================================================
    # Degree Methods (Shared Implementation)
    # =============================================================================
    
    def degree(self, node, order=None, size=None):
        """
        Calculate the degree of a node.

        Parameters
        ----------
        node : object
            The node to calculate degree for.
        order : int, optional
            The order of hyperedges to consider.
        size : int, optional
            The size of hyperedges to consider.

        Returns
        -------
        int
            The degree of the node.
        """
        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        """
        Calculate the degree sequence of the hypergraph.

        Parameters
        ----------
        order : int, optional
            The order of hyperedges to consider.
        size : int, optional
            The size of hyperedges to consider.

        Returns
        -------
        list
            The degree sequence.
        """
        return degree_sequence(self, order=order, size=size)

    # =============================================================================
    # Structural Information (Shared Implementation)
    # =============================================================================
    
    def num_nodes(self) -> int:
        """
        Returns the number of nodes in the hypergraph.

        Returns
        -------
        int
            Number of nodes in the hypergraph.
        """
        return len(self.get_nodes())

    def num_edges(self, *args, **kwargs) -> int:
        """
        Returns the number of edges in the hypergraph.
        
        Note: Parameters may vary by implementation for filtering.

        Returns
        -------
        int
            Number of edges in the hypergraph.
        """
        return len(self._edge_list)

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

    def max_size(self) -> int:
        """
        Returns the maximum size of the hypergraph.

        Returns
        -------
        int
            Maximum size of the hypergraph.
        """
        sizes = self.get_sizes()
        return max(sizes) if sizes else 0

    def max_order(self) -> int:
        """
        Returns the maximum order of the hypergraph.

        Returns
        -------
        int
            Maximum order of the hypergraph.
        """
        return self.max_size() - 1

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

    def distribution_sizes(self) -> Dict[int, int]:
        """
        Returns the distribution of sizes of the hyperedges in the hypergraph.

        Returns
        -------
        dict
            Distribution of sizes of the hyperedges in the hypergraph.
        """
        return dict(Counter(self.get_sizes()))

    def is_uniform(self) -> bool:
        """
        Check if the hypergraph is uniform, i.e. all hyperedges have the same size.

        Returns
        -------
        bool
            True if the hypergraph is uniform, False otherwise.
        """
        sizes = self.get_sizes()
        return len(set(sizes)) <= 1

    # =============================================================================
    # Metadata Management (Shared Implementation)
    # =============================================================================
    
    def get_hypergraph_metadata(self):
        """Get hypergraph metadata."""
        return self._hypergraph_metadata

    def set_hypergraph_metadata(self, metadata):
        """Set hypergraph metadata."""
        self._hypergraph_metadata = metadata

    def set_attr_to_hypergraph_metadata(self, field, value):
        """Set an attribute in hypergraph metadata."""
        self._hypergraph_metadata[field] = value

    def get_node_metadata(self, node):
        """Get metadata for a specific node."""
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self._node_metadata[node]

    def set_node_metadata(self, node, metadata):
        """Set metadata for a specific node."""
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node] = metadata

    def get_all_nodes_metadata(self):
        """Get metadata for all nodes."""
        return self._node_metadata

    def set_attr_to_node_metadata(self, node, field, value):
        """Set an attribute in node metadata."""
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        """Remove an attribute from node metadata."""
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        del self._node_metadata[node][field]

    @abstractmethod
    def get_edge_metadata(self, edge, *args, **kwargs):
        """
        Get metadata for a specific edge.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def set_edge_metadata(self, edge, *args, **kwargs):
        """
        Set metadata for a specific edge.
        
        Note: Signature varies by implementation.
        """
        pass

    def get_all_edges_metadata(self):
        """Get metadata for all edges."""
        return self._edge_metadata

    @abstractmethod
    def set_attr_to_edge_metadata(self, edge, *args, field, value):
        """
        Set an attribute in edge metadata.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def remove_attr_from_edge_metadata(self, edge, *args, field):
        """
        Remove an attribute from edge metadata.
        
        Note: Signature varies by implementation.
        """
        pass

    # =============================================================================
    # Utility Methods (Shared Implementation)
    # =============================================================================
    
    def isolated_nodes(self, size=None, order=None):
        """Get isolated nodes in the hypergraph."""
        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        """Check if a node is isolated."""
        return is_isolated(self, node, size=size, order=order)

    def clear(self):
        """Clear all data from the hypergraph."""
        self._edge_list.clear()
        self._weights.clear()
        self._hypergraph_metadata.clear()
        self._node_metadata.clear()
        self._edge_metadata.clear()
        self._reverse_edge_list.clear()
        self._incidences_metadata.clear()

    def copy(self):
        """
        Returns a copy of the hypergraph.

        Returns
        -------
        IHypergraph
            A copy of the hypergraph.
        """
        return copy.deepcopy(self)

    def __str__(self):
        """
        Returns a string representation of the hypergraph.

        Returns
        -------
        str
            A string representation of the hypergraph.
        """
        title = "Hypergraph with {} nodes and {} edges.\n".format(
            self.num_nodes(), self.num_edges()
        )
        details = "Distribution of hyperedge sizes: {}".format(
            self.distribution_sizes()
        )
        return title + details

    def __len__(self):
        """
        Returns the number of edges in the hypergraph.

        Returns
        -------
        int
            The number of edges in the hypergraph.
        """
        return len(self._edge_list)

    def __iter__(self):
        """
        Returns an iterator over the edges in the hypergraph.

        Returns
        -------
        iterator
            An iterator over the edges in the hypergraph.
        """
        return iter(self._edge_list.items())

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

    def get_mapping(self):
        """
        Map the nodes of the hypergraph to integers in [0, n_nodes).

        Returns
        -------
        LabelEncoder
            The mapping.
        """
        encoder = LabelEncoder()
        encoder.fit(self.get_nodes())
        return encoder