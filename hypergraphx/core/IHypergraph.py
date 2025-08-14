from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import copy
from sklearn.preprocessing import LabelEncoder
from collections import Counter


class IHypergraph(ABC):
    """
    Abstract base class defining the common interface for all hypergraph implementations.
    
    This class specifies the required properties and methods that must be implemented
    by all hypergraph types: Hypergraph, TemporalHypergraph, MultiplexHypergraph, and DirectedHypergraph.

    Contains functionality common to both directed and undirected hypergraphs.
    """

    def __init__(
        self,
        edge_list: Optional[List]=None,
        weighted: bool = False,
        weights:Optional[List[int]]=None,
        hypergraph_metadata: Optional[Dict] = None,
        node_metadata: Optional[Dict] = None,
        edge_metadata: Optional[List[Dict]] = None
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
        self._hypergraph_metadata = hypergraph_metadata or dict()
        self._hypergraph_metadata.update({"weighted": weighted})

        self._weighted:bool = weighted
        self._weights:dict = dict()
        self._node_metadata:Dict[Any, Dict] = node_metadata or dict()
        self._edge_metadata:Dict[Tuple, Dict] = {
            edge_list[i]: edge_metadata[i]
            for i in range(len(edge_list))
        } if edge_metadata and edge_list else dict()
        
        # store _edge_list and _reverse_edge_list as dictionaries
        #   keys of _edge_list are edges
        #   values of _edge_list are edge id's, ie integers
        self._edge_list:dict = dict()
        self._reverse_edge_list:dict = dict()
        self._next_edge_id:int = 0

        self._incidences_metadata = {}

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
                  metadata: Optional[List | Dict] = None) -> None:
        """
        Add a list of nodes to the hypergraph.

        Parameters
        ----------
        node_list : list
            The list of nodes to add.
        metadata : list or dict
            The list of nodes' metadata to add.

        Returns
        -------
        None
        """
        if metadata is None:
            metadata = {}
        # if the metadata was provided in list form, convert it to a dict
        if isinstance(metadata, List):
            if len(node_list) == len(metadata):
                metadata = {
                    node_list[i]: metadata[i]
                    for i in range(len(node_list))
                }
            else:
                raise ValueError(f"len({node_list}) != len({metadata})")
        
        for node in node_list:
            self.add_node(node, metadata=metadata.get(node))

    @abstractmethod
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
        pass

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

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any of the nodes is not in the hypergraph.
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

    def add_edges(self,
                  edge_list:List[Tuple[Tuple, Tuple]],
                  
                  weights:List[int]=None,
                  metadata:List[Dict]=None,
                  *args,
                  **kwargs) -> None:
        """Add a list of hyperedges to the hypergraph. If a hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge_list : list
            The list of hyperedges to add.
        edge_layer : list
            The list of layers to which the hyperedges belong.
        weights : list, optional
            The list of weights of the hyperedges. If the hypergraph is weighted, this must be provided.
        metadata : list, optional
            The list of metadata of the hyperedges.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the hypergraph is weighted and no weights are provided or if the hypergraph is not weighted and weights are provided.
        """
        if weights is not None and not self._weighted:
            print(
                "Warning: weights are provided but the hypergraph is not weighted. The hypergraph will be weighted."
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
            self.add_edge(
                edge=edge,
                weight=(
                    weights[i] if self._weighted and weights is not None else None
                ),
                metadata=metadata[i] if metadata is not None else None,
                *args,
                **kwargs,
            )

    @abstractmethod
    def remove_edge(self, edge, *args, **kwargs) -> None:
        """
        Remove an edge from the hypergraph.
        
        Note: Signature varies by implementation.
        """
        pass

    @abstractmethod
    def remove_edges(self, edge_list) -> None:
        """
        Remove multiple edges from the hypergraph.

        Parameters
        ----------
        edge_list : list
            The list of edges to remove.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any edge is not in the hypergraph.
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
    def get_neighbors(self, node, order: int = None, size: int = None):
        """
        Get the neighbors of a node in the hypergraph.

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
        set
            The neighbors of the node.

        Raises
        ------
        ValueError
            If order and size are both specified or neither are specified.
        """
        pass

    @abstractmethod
    def get_incident_edges(self, node, order: int = None, size: int = None) -> List[Tuple]:
        """
        Get the incident edges of a node.

        Parameters
        ----------
        node : object
            The node of interest.
        order : int, optional
            The order of the hyperedges to consider. If None, all hyperedges are considered.
        size : int, optional
            The size of the hyperedges to consider. If None, all hyperedges are considered.

        Returns
        -------
        list
            The list of incident edges.
        """
        pass

    def check_edge(self, edge, *args, **kwargs) -> bool:
        """
        Check if an edge exists in the hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to check.

        Returns
        -------
        bool
            True if the edge is in the hypergraph, False otherwise.

        """
        edge = self._canon_edge(edge)
        k = self._restructure_query_edge(edge, *args, **kwargs)
        return k in self._edge_list

    def get_edge_list(self) -> Dict[Tuple, int]:
        """Get the edge list dictionary."""
        return self._edge_list

    def set_edge_list(self, edge_list: List[Tuple]):
        """Set the edge list dictionary."""
        self._edge_list = {
            e: i for i, e in enumerate(edge_list)
        }
        self._next_edge_id = len(edge_list)

    # =============================================================================
    # Weight Management
    # =============================================================================
    
    def get_weight(self, edge, *args, **kwargs):
        """Returns the weight of the specified edge.

        Parameters
        ----------
        edge : tuple
            The edge to get the weight of.

        Returns
        -------
        float
            Weight of the specified edge.
        """
        edge = self._canon_edge(edge)
        k = self._restructure_query_edge(edge, *args, **kwargs)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(k))
        else:
            return self._weights[self._edge_list[k]]

    def set_weight(self, edge, weight, *args, **kwargs) -> None:
        """Sets the weight of the specified edge.

        Parameters
        ----------
        edge : tuple
            The edge to set the weight of.

        weight : float
            The weight to set.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the edge is not in the hypergraph.
        """
        if not self._weighted and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )

        edge = self._canon_edge(edge)
        k = self._restructure_query_edge(edge, *args, **kwargs)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        else:
            self._weights[self._edge_list[k]] = weight

    @abstractmethod
    def get_weights(self, order=None, size=None, up_to=False, asdict=False):
        """Returns the list of weights of the edges in the hypergraph. If order is specified, it returns the list of weights of the edges of the specified order.
        If size is specified, it returns the list of weights of the edges of the specified size. If both order and size are specified, it raises a ValueError.
        If up_to is True, it returns the list of weights of the edges of order smaller or equal to the specified order.

        Parameters
        ----------
        order : int, optional
            Order of the edges to get the weights of.

        size : int, optional
            Size of the edges to get the weights of.

        up_to : bool, optional
            If True, it returns the list of weights of the edges of order smaller or equal to the specified order. Default is False.

        Returns
        -------
        list
            List of weights of the edges in the hypergraph.

        Raises
        ------
        ValueError
            If both order and size are specified.

        """
        pass

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

    @abstractmethod
    def num_edges(self) -> int:
        """Returns the number of edges in the hypergraph.

        Returns
        -------
        int
            Number of edges in the hypergraph.
        """
        pass

    @abstractmethod
    def get_sizes(self) -> List[int]:
        """Returns the list of sizes of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of sizes of the hyperedges in the hypergraph.

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

    def get_orders(self) -> List[int]:
        """
        Get the order of each edge in the hypergraph.

        Returns
        -------
        list
            A list of integers representing the order of each edge.
        """
        return [size - 1 for size in self.get_sizes()]

    def distribution_sizes(self) -> Dict[int, int]:
        """
        Returns the distribution of sizes of the hyperedges in the hypergraph.

        Returns
        -------
        dict
            Distribution of sizes of the hyperedges in the hypergraph.
        """
        return dict(Counter(self.get_sizes()))

    @abstractmethod
    def is_uniform(self) -> bool:
        """
        Check if the hypergraph is uniform, i.e. all hyperedges have the same size.

        Returns
        -------
        bool
            True if the hypergraph is uniform, False otherwise.
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
    # Metadata Management (Shared Implementation)
    # =============================================================================
    
    # Hypergraph metadata
    def get_hypergraph_metadata(self):
        """Get hypergraph metadata."""
        return self._hypergraph_metadata

    def set_hypergraph_metadata(self, metadata):
        """Set hypergraph metadata."""
        self._hypergraph_metadata = metadata

    def set_attr_to_hypergraph_metadata(self, field, value):
        """Set an attribute in hypergraph metadata."""
        self._hypergraph_metadata[field] = value

    def add_attr_to_node_metadata(self, field, value):
        """Included for backwards compatibility"""
        self.set_attr_to_hypergraph_metadata(field, value)

    # Node metadata
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

    def add_attr_to_node_metadata(self, node, field, value):
        """Included for backwards compatibility"""
        self.set_attr_to_node_metadata(node, field, value)

    def remove_attr_from_node_metadata(self, node, field):
        """Remove an attribute from node metadata."""
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        del self._node_metadata[node][field]

    # Edge metadata    
    def get_edge_metadata(self, edge, *args, **kwargs) -> dict:
        """
        Get metadata for a specific edge.
        """
        edge = self._canon_edge(edge)
        k = self._restructure_query_edge(edge, *args, **kwargs)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return dict(self._edge_metadata[k])
    
    def set_edge_metadata(self, edge, metadata:Dict, *args, **kwargs):
        edge = self._canon_edge(edge)
        k = self._restructure_query_edge(edge, *args, **kwargs)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._edge_metadata[k] = metadata
    
    def get_all_edges_metadata(self) -> Dict[Tuple, Dict]:
        """Get metadata for all edges."""
        return self._edge_metadata
    
    def set_attr_to_edge_metadata(self, edge, field, value, *args, **kwargs):
        edge = self._canon_edge(edge)
        k = self._restructure_query_edge(edge, *args, **kwargs)
        if k not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._edge_metadata[k][field] = value
    
    def add_attr_to_edge_metadata(self, edge, field, value, *args, **kwargs):
        """Included for backwards compatibility"""
        self.set_attr_to_edge_metadata(edge, field, value, *args, **kwargs)

    def remove_attr_from_edge_metadata(self, edge, field, *args, **kwargs):
        edge = self._canon_edge(edge)
        k = self._restructure_query_edge(edge, *args, **kwargs)
        if k not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        del self._edge_metadata[k][field]
    
    # Incidence metadata
    @abstractmethod
    def get_incidence_metadata(self, edge, node):
        """Get incidence metadata for a specific edge-node pair."""
        pass
    
    @abstractmethod
    def set_incidence_metadata(self, edge, node, metadata):
        """Set incidence metadata for a specific edge-node pair."""
        pass

    @abstractmethod
    def get_all_incidences_metadata(self):
        """Get all incidence metadata."""
        pass

    @abstractmethod
    def _restructure_query_edge(self, k: Tuple[Tuple, Any]):
        """
        An implementation-specific helper for modifying a query edge
        prior to metadata retrieval.
        """
        pass

    # =============================================================================
    # Utility Methods (Shared Implementation)
    # =============================================================================
    
    @abstractmethod
    def _canon_edge(self, edge: Tuple) -> Tuple:
        """
        Gets the canonical form of an edge (sorts the inner tuples)
        Works for hyperedges but WILL BREAK FOR METAEDGES
        TODO: Add recursive canonicalization for future metagraph integration
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all data from the hypergraph."""
        pass

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
    def expose_attributes_for_hashing(self) -> dict:
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