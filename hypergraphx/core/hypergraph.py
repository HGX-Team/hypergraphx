import copy
import warnings
from typing import Tuple, Any, List, Dict, Optional

from sklearn.preprocessing import LabelEncoder

from hypergraphx.core.IUndirectedHypergraph import IUndirectedHypergraph


class Hypergraph(IUndirectedHypergraph):
    """
    A Hypergraph is a generalization of a graph where an edge (hyperedge) can connect
    any number of nodes. It is represented as a set of nodes and a set of hyperedges,
    where each hyperedge is a subset of nodes.
    
    This implementation now inherits from IUndirectedHypergraph to leverage common functionality.
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
        # Call parent constructor (IUndirectedHypergraph)
        super().__init__(edge_list, weighted, weights, hypergraph_metadata, node_metadata, edge_metadata)
        
        # Set hypergraph type in metadata
        self._hypergraph_metadata.update({"type": "Hypergraph"})

        # Initialize Hypergraph-specific attributes
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
    # Implementation of Abstract Methods from IUndirectedHypergraph
    # =============================================================================

    def _add_edge_implementation(self, edge, weight, metadata, **kwargs):
        """
        Implementation of abstract method for adding a single edge.
        
        Parameters
        ----------
        edge : tuple
            The edge to add.
        weight : float, optional
            The weight of the edge.
        metadata : dict, optional
            The metadata of the edge.
        **kwargs
            Additional parameters (unused for Hypergraph).
        """
        self.add_edge(edge, weight=weight, metadata=metadata)

    def _extract_nodes_from_edge(self, edge) -> List:
        """
        Extract node list from an edge representation.
        For Hypergraph, edges are simple tuples of nodes.
        
        Parameters
        ----------
        edge : tuple
            The edge representation (tuple of nodes).
            
        Returns
        -------
        list
            List of nodes in the edge.
        """
        return list(edge)

    def _get_edge_size(self, edge_key) -> int:
        """
        Get the size of an edge given its key representation.
        For Hypergraph, edge keys are tuples of nodes.
        
        Parameters
        ----------
        edge_key : tuple
            The edge key representation (tuple of nodes).
            
        Returns
        -------
        int
            Size of the edge.
        """
        return len(edge_key)

    # =============================================================================
    # Hypergraph-Specific Node Management Implementation
    # =============================================================================

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

    # =============================================================================
    # Hypergraph-Specific Edge Management Implementation
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
    # Hypergraph-Specific Utility Methods
    # =============================================================================

    def _restructure_query_edge(self, k: Tuple[Tuple, Any]):
        """
        An implementation-specific helper for modifying a query edge
        prior to metadata retrieval.
        """
        return tuple(sorted(k))

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
    # Projections and Transformations
    # =============================================================================

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
        super().clear()  # Calls IUndirectedHypergraph.clear() which calls IHypergraph.clear()
        self._empty_edges.clear()

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
        base_data = super().expose_data_structures()
        base_data.update({
            "type": "Hypergraph",
            "empty_edges": self._empty_edges,
        })
        return base_data

    def populate_from_dict(self, data):
        """
        Populate the attributes of the hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        super().populate_from_dict(data)
        self._empty_edges = data.get("empty_edges", {})

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