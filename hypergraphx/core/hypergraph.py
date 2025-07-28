import copy
import warnings
from typing import Tuple, Any, List, Dict, Optional

from sklearn.preprocessing import LabelEncoder

from hypergraphx.core.i_hypergraph import IHypergraph

class Hypergraph(IHypergraph):
    """
    A Hypergraph is a generalization of a graph where an edge (hyperedge) can connect
    any number of nodes. It is represented as a set of nodes and a set of hyperedges,
    where each hyperedge is a subset of nodes.
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
        super().__init__(edge_list, weighted, weights, hypergraph_metadata, node_metadata, edge_metadata)
        
        # Set hypergraph type in metadata
        self._hypergraph_metadata.update({"type": "Hypergraph"})

        # Initialize adjacency list specific to undirected hypergraphs
        self._adj = {}
        self._empty_edges = {}

        # Add node metadata if provided (using parent's add_nodes method)
        if node_metadata:
            self.add_nodes(
                list(node_metadata.keys()),
                metadata=node_metadata
            )

        # Add edges if provided
        if edge_list:
            if weighted and weights is not None and len(edge_list) != len(weights):
                raise ValueError("Edge list and weights must have the same length.")
            self.add_edges(
                edge_list,
                weights=weights,
                metadata=edge_metadata
            )

    # =============================================================================
    # Node Management Implementation
    # =============================================================================
    
    def add_node(self, node, metadata=None):
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
        # Call parent implementation for metadata handling
        super().add_node(node, metadata)
        
        # Add to adjacency list if not already present
        if node not in self._adj:
            self._adj[node] = []

    def remove_node(self, node, keep_edges=False):
        """Remove a node from the hypergraph.

        Parameters
        ----------
        node
            The node to remove.
        keep_edges : bool, optional
            If True, the edges incident to the node are kept, but the node is removed from the edges. 
            If False, the edges incident to the node are removed. Default is False.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the node is not in the hypergraph.
        """
        if node not in self._adj:
            raise KeyError("Node {} not in hypergraph.".format(node))
            
        if not keep_edges:
            self.remove_edges(
                [self._reverse_edge_list[edge_id] for edge_id in self._adj[node]]
            )
        else:
            to_remove = []
            for edge_id in self._adj[node]:
                edge = self._reverse_edge_list[edge_id]
                self.add_edge(
                    tuple(sorted([n for n in edge if n != node])),
                    weight=self.get_weight(edge),
                    metadata=self.get_edge_metadata(edge),
                )
                to_remove.append(edge)
            self.remove_edges(to_remove)
            
        del self._adj[node]
        # Remove from parent's node metadata
        if node in self._node_metadata:
            del self._node_metadata[node]

    def remove_nodes(self, node_list, keep_edges=False):
        """
        Remove a list of nodes from the hypergraph.

        Parameters
        ----------
        node_list : list
            The list of nodes to remove.
        keep_edges : bool, optional
            If True, the edges incident to the nodes are kept, but the nodes are removed from the edges. 
            If False, the edges incident to the nodes are removed. Default is False.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any of the nodes is not in the hypergraph.
        """
        for node in node_list:
            self.remove_node(node, keep_edges=keep_edges)

    def get_nodes(self, metadata=False):
        """
        Returns the list of nodes in the hypergraph. If metadata is True, it returns a dictionary of node metadata.

        Parameters
        ----------
        metadata : bool, optional
            If True, return node metadata dictionary. If False, return list of nodes.

        Returns
        -------
        list or dict
            List of nodes in the hypergraph. If metadata is True, it returns a dictionary of node metadata.
        """
        if not metadata:
            return list(self._adj.keys())
        else:
            return {node: self.get_node_metadata(node) for node in self._adj.keys()}

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
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            neigh = set()
            edges = self.get_incident_edges(node)
            for edge in edges:
                neigh.update(edge)
            if node in neigh:
                neigh.remove(node)
            return neigh
        else:
            if order is None:
                order = size - 1
            neigh = set()
            edges = self.get_incident_edges(node, order=order)
            for edge in edges:
                neigh.update(edge)
            if node in neigh:
                neigh.remove(node)
            return neigh

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

    # =============================================================================
    # Edge Management Implementation
    # =============================================================================
    
    def add_edge(self, edge, weight=None, metadata=None):
        """Add a hyperedge to the hypergraph. If the hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge : tuple
            The hyperedge to add.
        weight : float, optional
            The weight of the hyperedge. If the hypergraph is weighted, this must be provided.
        metadata : dict, optional
            The metadata of the hyperedge.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the hypergraph is weighted and no weight is provided or if the hypergraph is not weighted and a weight is provided.
        """
        if not self._weighted and weight is not None and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )

        if weight is None:
            weight = 1

        edge = self._canon_edge(edge)
        if metadata is None:
            metadata = {}

        if edge not in self._edge_list:
            self._edge_list[edge] = self._next_edge_id
            self._reverse_edge_list[self._next_edge_id] = edge
            self._weights[self._next_edge_id] = 1 if not self._weighted else weight
            self._next_edge_id += 1
        elif edge in self._edge_list and self._weighted:
            self._weights[self._edge_list[edge]] += weight

        # Set edge metadata using parent method
        if edge in self._edge_list:
            self.set_edge_metadata(edge, metadata)

        # Update adjacency list
        for node in edge:
            self.add_node(node)
            if self._edge_list[edge] not in self._adj[node]:
                self._adj[node].append(self._edge_list[edge])

    def add_edges(self, edge_list, weights=None, metadata=None):
        """Add a list of hyperedges to the hypergraph. If a hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge_list : list
            The list of hyperedges to add.
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

        i = 0
        if edge_list is not None:
            for edge in edge_list:
                self.add_edge(
                    edge,
                    weight=(
                        weights[i] if self._weighted and weights is not None else None
                    ),
                    metadata=metadata[i] if metadata is not None else None,
                )
                i += 1

    def remove_edge(self, edge):
        """Remove an edge from the hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to remove.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the edge is not in the hypergraph.
        """
        edge = self._canon_edge(edge)
        if edge not in self._edge_list:
            raise KeyError("Edge {} not in hypergraph.".format(edge))

        edge_id = self._edge_list[edge]

        del self._reverse_edge_list[edge_id]
        if edge_id in self._weights:
            del self._weights[edge_id]
        if edge in self._edge_metadata:
            del self._edge_metadata[edge]
        
        # Remove from adjacency lists
        for node in edge:
            if node in self._adj and edge_id in self._adj[node]:
                self._adj[node].remove(edge_id)

        # Remove from the edge list
        del self._edge_list[edge]

    def remove_edges(self, edge_list):
        """
        Remove a list of edges from the hypergraph.

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
        """
        for edge in edge_list:
            self.remove_edge(edge)

    def get_edges(
        self,
        order=None,
        size=None,
        up_to=False,
        subhypergraph=False,
        keep_isolated_nodes=False,
        metadata=False,
    ):
        """Get edges from the hypergraph with various filtering options."""
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if not subhypergraph and keep_isolated_nodes:
            raise ValueError("Cannot keep nodes if not returning subhypergraphs.")

        if order is None and size is None:
            edges = list(self._edge_list.keys())
        else:
            if size is not None:
                order = size - 1
            if not up_to:
                edges = [
                    edge
                    for edge in list(self._edge_list.keys())
                    if len(edge) - 1 == order
                ]
            else:
                edges = [
                    edge
                    for edge in list(self._edge_list.keys())
                    if len(edge) - 1 <= order
                ]

        edge_metadata = [self.get_edge_metadata(edge) for edge in edges]
        edge_weights = [self.get_weight(edge) for edge in edges] if self._weighted else None
        if subhypergraph and keep_isolated_nodes:
            h = Hypergraph(weighted=self._weighted)
            nodes = list(self.get_nodes())
            node_metadata = [self.get_node_metadata(node) for node in nodes]
            h.add_nodes(nodes, metadata=node_metadata)
            h.add_edges(edges, weights=edge_weights, metadata=edge_metadata)
            return h
        
        elif subhypergraph:
            h = Hypergraph(weighted=self._weighted)
            h.add_edges(edges, weights=edge_weights, metadata=edge_metadata)
            return h
        
        else:
            return (
                edges
                if not metadata
                else {edge: self.get_edge_metadata(edge) for edge in edges}
            )

    # =============================================================================
    # Weight Management (Inherited methods work, but we can add specific implementations)
    # =============================================================================

    def get_weights(self, order=None, size=None, up_to=False, asdict=False):
        """Returns the list of weights of the edges in the hypergraph."""
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
            
        if order is None and size is None:
            w = {
                edge: self._weights[self._edge_list[edge]] for edge in self.get_edges()
            }
        else:
            if size is not None:
                order = size - 1
            w = {
                edge: self._weights[self._edge_list[edge]]
                for edge in self.get_edges(order=order, up_to=up_to)
            }

        if asdict:
            return w
        else:
            return list(w.values())

    # =============================================================================
    # Information Methods
    # =============================================================================

    def num_edges(self, order=None, size=None, up_to=False):
        """Returns the number of edges in the hypergraph."""
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")

        if order is None and size is None:
            return len(self._edge_list)
        else:
            if size is not None:
                order = size - 1
            if not up_to:
                return sum(1 for edge in self._edge_list if len(edge) - 1 == order)
            else:
                return sum(1 for edge in self._edge_list if len(edge) - 1 <= order)

    def get_sizes(self):
        """Returns the list of sizes of the hyperedges in the hypergraph."""
        return [len(edge) for edge in self._edge_list.keys()]

    def is_uniform(self):
        """
        Check if the hypergraph is uniform, i.e. all hyperedges have the same size.

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
    # Utility Methods
    # =============================================================================

    def _canon_edge(self, edge: Tuple) -> Tuple:
        """
        Gets the canonical form of an edge (sorts the nodes in the edge).
        """
        return tuple(sorted(edge))

    def _restructure_query_edge(self, k: Tuple[Tuple, Any]):
        """
        An implementation-specific helper for modifying a query edge
        prior to metadata retrieval.
        """
        return tuple(sorted(k))

    def get_adj_dict(self):
        """Get the adjacency dictionary."""
        return self._adj

    def set_adj_dict(self, adj):
        """Set the adjacency dictionary."""
        self._adj = adj

    def add_empty_edge(self, name, metadata):
        """Add an empty edge with metadata."""
        if name not in self._empty_edges:
            self._empty_edges[name] = metadata
        else:
            raise ValueError("Edge {} already in hypergraph.".format(name))

    # =============================================================================
    # Subgraph Methods
    # =============================================================================

    def subhypergraph(self, nodes: list):
        """
        Return a subhypergraph induced by the nodes in the list.

        Parameters
        ----------
        nodes : list
            List of nodes to be included in the subhypergraph.

        Returns
        -------
        Hypergraph
            Subhypergraph induced by the nodes in the list.
        """
        h = Hypergraph(weighted=self._weighted)
        h.add_nodes(nodes)
        for node in nodes:
            h.set_node_metadata(node, self.get_node_metadata(node))
        for edge in self._edge_list:
            if set(edge).issubset(set(nodes)):
                if self._weighted:
                    h.add_edge(
                        edge,
                        weight=self.get_weight(edge),
                        metadata=self.get_edge_metadata(edge),
                    )
                else:
                    h.add_edge(edge, metadata=self.get_edge_metadata(edge))
        return h

    def subhypergraph_by_orders(
        self, orders: list = None, sizes: list = None, keep_nodes=True
    ):
        """Return a subhypergraph induced by the edges of the specified orders."""
        if orders is None and sizes is None:
            raise ValueError(
                "At least one between orders and sizes should be specified"
            )
        if orders is not None and sizes is not None:
            raise ValueError("Order and size cannot be both specified.")
            
        h = Hypergraph(weighted=self.is_weighted())
        if keep_nodes:
            h.add_nodes(node_list=list(self.get_nodes()))
            for node in self.get_nodes():
                h.set_node_metadata(node, self.get_node_metadata(node))

        if sizes is None:
            sizes = [order + 1 for order in orders]

        for size in sizes:
            edges = self.get_edges(size=size)
            for edge in edges:
                if h.is_weighted():
                    h.add_edge(
                        edge,
                        self.get_weight(edge),
                        self.get_edge_metadata(edge)
                    )
                else:
                    h.add_edge(
                        edge,
                        metadata=self.get_edge_metadata(edge)
                    )

        return h

    def subhypergraph_largest_component(self, size=None, order=None):
        """
        Returns a subhypergraph induced by the nodes in the largest component of the hypergraph.
        """
        nodes = self.largest_component(size=size, order=order)
        return self.subhypergraph(nodes)

    # =============================================================================
    # Analysis Methods (delegated to external modules)
    # =============================================================================

    def degree(self, node, order=None, size=None):
        from hypergraphx.measures.degree import degree
        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_sequence
        return degree_sequence(self, order=order, size=size)

    def degree_distribution(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_distribution
        return degree_distribution(self, order=order, size=size)

    # Connected Components
    def is_connected(self, size=None, order=None):
        from hypergraphx.utils.cc import is_connected
        return is_connected(self, size=size, order=order)

    def connected_components(self, size=None, order=None):
        from hypergraphx.utils.cc import connected_components
        return connected_components(self, size=size, order=order)

    def node_connected_component(self, node, size=None, order=None):
        from hypergraphx.utils.cc import node_connected_component
        return node_connected_component(self, node, size=size, order=order)

    def num_connected_components(self, size=None, order=None):
        from hypergraphx.utils.cc import num_connected_components
        return num_connected_components(self, size=size, order=order)

    def largest_component(self, size=None, order=None):
        from hypergraphx.utils.cc import largest_component
        return largest_component(self, size=size, order=order)

    def largest_component_size(self, size=None, order=None):
        from hypergraphx.utils.cc import largest_component_size
        return largest_component_size(self, size=size, order=order)

    # Matrix operations
    def binary_incidence_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import binary_incidence_matrix
        return binary_incidence_matrix(self, return_mapping)

    def incidence_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import incidence_matrix
        return incidence_matrix(self, return_mapping)

    def adjacency_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import adjacency_matrix
        return adjacency_matrix(self, return_mapping)

    # Utils
    def isolated_nodes(self, size=None, order=None):
        from hypergraphx.utils.cc import isolated_nodes
        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.cc import is_isolated
        return is_isolated(self, node, size=size, order=order)

    def dual_random_walk_adjacency(self, return_mapping: bool = False):
        from hypergraphx.linalg import dual_random_walk_adjacency
        return dual_random_walk_adjacency(self, return_mapping)

    def adjacency_factor(self, t: int = 0):
        from hypergraphx.linalg import adjacency_factor
        return adjacency_factor(self, t)

    def to_line_graph(self, distance="intersection", s: int = 1, weighted=False):
        from hypergraphx.representations.projections import line_graph
        return line_graph(self, distance, s, weighted)

    # =============================================================================
    # Incidence Metadata (specific implementation for undirected hypergraphs)
    # =============================================================================

    def set_incidence_metadata(self, edge, node, metadata):
        """Set incidence metadata for a specific edge-node pair."""
        edge = self._canon_edge(edge)
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._incidences_metadata[(edge, node)] = metadata

    def get_incidence_metadata(self, edge, node):
        """Get incidence metadata for a specific edge-node pair."""
        edge = self._canon_edge(edge)
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self._incidences_metadata.get((edge, node), {})

    def get_all_incidences_metadata(self):
        """Get all incidence metadata."""
        return {k: v for k, v in self._incidences_metadata.items()}

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def clear(self):
        """Clear all data from the hypergraph."""
        self._edge_list.clear()
        self._adj.clear()
        self._weights.clear()
        self._hypergraph_metadata.clear()
        self._incidences_metadata.clear()
        self._node_metadata.clear()
        self._edge_metadata.clear()
        self._empty_edges.clear()
        self._reverse_edge_list.clear()
        self._next_edge_id = 0

    # =============================================================================
    # Serialization Support
    # =============================================================================

    def expose_data_structures(self):
        """
        Expose the internal data structures of the hypergraph for serialization.

        Returns
        -------
        dict
            A dictionary containing all internal attributes of the hypergraph.
        """
        return {
            "type": "Hypergraph",
            "_weighted": self._weighted,
            "_adj": self._adj,
            "_edge_list": self._edge_list,
            "_weights": self._weights,
            "hypergraph_metadata": self._hypergraph_metadata,
            "node_metadata": self._node_metadata,
            "edge_metadata": self._edge_metadata,
            "incidences_metadata": self._incidences_metadata,
            "empty_edges": self._empty_edges,
            "reverse_edge_list": self._reverse_edge_list,
            "next_edge_id": self._next_edge_id,
        }

    def populate_from_dict(self, data):
        """
        Populate the attributes of the hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        self._weighted = data.get("_weighted", False)
        self._adj = data.get("_adj", {})
        self._edge_list = data.get("_edge_list", {})
        self._weights = data.get("_weights", {})
        self._hypergraph_metadata = data.get("hypergraph_metadata", {})
        self._incidences_metadata = data.get("incidences_metadata", {})
        self._node_metadata = data.get("node_metadata", {})
        self._edge_metadata = data.get("edge_metadata", {})
        self._empty_edges = data.get("empty_edges", {})
        self._reverse_edge_list = data.get("reverse_edge_list", {})
        self._next_edge_id = data.get("next_edge_id", 0)

    def expose_attributes_for_hashing(self):
        """
        Expose relevant attributes for hashing specific to Hypergraph.

        Returns
        -------
        dict
            A dictionary containing key attributes.
        """
        edges = []
        for edge in sorted(self._edge_list.keys()):
            sorted_edge = sorted(edge)
            edge_id = self._edge_list[edge]
            edges.append(
                {
                    "nodes": sorted_edge,
                    "weight": self._weights.get(edge_id, 1),
                    "metadata": self.get_edge_metadata(edge)
                }
            )

        nodes = []
        for node in sorted(self._adj.keys()):
            nodes.append({"node": node, "metadata": self._node_metadata[node]})

        return {
            "type": "Hypergraph",
            "weighted": self._weighted,
            "hypergraph_metadata": self._hypergraph_metadata,
            "edges": edges,
            "nodes": nodes,
        }