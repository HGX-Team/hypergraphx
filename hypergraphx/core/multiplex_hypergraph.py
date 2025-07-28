from typing import Tuple, Any, List, Dict, Set, Optional, Union
from collections import Counter

from hypergraphx.core.i_hypergraph import IHypergraph
from hypergraphx import Hypergraph


class MultiplexHypergraph(IHypergraph):
    """
    A Multiplex Hypergraph is a hypergraph where hyperedges are organized into multiple layers.
    Each layer shares the same node-set and represents a specific context or relationship between nodes, and hyperedges can
    have weights and metadata specific to their layer.
    """

    def __init__(
        self,
        edge_list: Optional[List]=None,
        edge_layer: Optional[List]=None,
        weighted: bool = False,
        weights: Optional[List[int]]=None,
        hypergraph_metadata: Optional[Dict] = None,
        node_metadata: Optional[Dict] = None,
        edge_metadata: Optional[List[Dict]] = None
    ):
        """
        Initialize a Multiplex Hypergraph with optional edges, layers, weights, and metadata.

        Parameters
        ----------
        edge_list : list of tuples, optional
            A list of edges where each edge is represented as a tuple of nodes.
            If `edge_layer` is not provided, each tuple in `edge_list` should have
            the format `(edge, layer)`, where `edge` is itself a tuple of nodes.
        edge_layer : list of str, optional
            A list of layer names corresponding to each edge in `edge_list`.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is False.
        weights : list of float, optional
            A list of weights for each edge in `edge_list`. Must be provided if `weighted` is True.
        hypergraph_metadata : dict, optional
            Metadata for the hypergraph as a whole. Default is an empty dictionary.
        node_metadata : dict, optional
            A dictionary of metadata for nodes, where keys are node identifiers and values are metadata dictionaries.
        edge_metadata : list of dict, optional
            A list of metadata dictionaries for each edge in `edge_list`.

        Raises
        ------
        ValueError
            If `edge_list` and `edge_layer` have mismatched lengths.
            If `edge_list` contains improperly formatted edges when `edge_layer` is None.
        """
        # Call parent constructor
        super().__init__(
            edge_list=None,  # We'll handle edge_list ourselves
            weighted=weighted,
            weights=None,    # We'll handle weights ourselves
            hypergraph_metadata=hypergraph_metadata,
            node_metadata=node_metadata,
            edge_metadata=None  # We'll handle edge_metadata ourselves
        )
        
        # Update hypergraph metadata with multiplex-specific info
        self._hypergraph_metadata.update({"type": "MultiplexHypergraph"})
        
        # Initialize multiplex-specific attributes
        self._adj = {}
        self._existing_layers = set()

        # Handle edge and layer consistency
        if edge_list is not None and edge_layer is None:
            # Extract layers from edge_list if layer information is embedded
            if all(isinstance(edge, tuple) and len(edge) == 2 for edge in edge_list):
                edge_layer = [edge[1] for edge in edge_list]
                edge_list = [edge[0] for edge in edge_list]
            else:
                raise ValueError(
                    "If edge_layer is not provided, edge_list must contain tuples of the form (edge, layer)."
                )

        if edge_list is not None:
            if edge_layer is not None and len(edge_list) != len(edge_layer):
                raise ValueError("Edge list and edge layer must have the same length.")
            self.add_edges(
                edge_list,
                edge_layer=edge_layer,
                weights=weights,
                metadata=edge_metadata,
            )

    # =============================================================================
    # Node Management (Implementation of abstract methods)
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

        Returns
        -------
        None
        """
        super().add_node(node, metadata)
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
            return self._node_metadata
        else:
            return list(self._node_metadata.keys())

    def remove_node(self, node: Any, keep_edges: bool = False) -> None:
        """
        Remove a node from the multiplex hypergraph.

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
        if node not in self._adj:
            raise ValueError(f"Node {node} not in hypergraph.")

        edges_to_process = list(self._adj[node])

        if keep_edges:
            for edge_id in edges_to_process:
                edge, layer = self._reverse_edge_list[edge_id]
                updated_edge = tuple(n for n in edge if n != node)

                # Get current metadata and weight before removing
                current_weight = self._weights.get(edge_id, 1)
                current_metadata = self.get_edge_metadata(edge, layer)
                
                self.remove_edge((edge, layer))
                if updated_edge:
                    self.add_edge(
                        updated_edge,
                        layer,
                        weight=current_weight,
                        metadata=current_metadata,
                    )
        else:
            for edge_id in edges_to_process:
                edge, layer = self._reverse_edge_list[edge_id]
                self.remove_edge((edge, layer))

        del self._adj[node]
        if node in self._node_metadata:
            del self._node_metadata[node]

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
        for node in node_list:
            self.remove_node(node, keep_edges=keep_edges)

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
        return node in self._node_metadata

    # =============================================================================
    # Edge Management (Implementation of abstract methods)
    # =============================================================================

    def add_edge(self, edge, layer, weight=None, metadata=None) -> None:
        """Add a hyperedge to the hypergraph. If the hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge : tuple
            The hyperedge to add.
        layer : str
            The layer to which the hyperedge belongs.
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
        if weight is None:
            weight = 1

        if not self._weighted and weight is not None and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )

        self._existing_layers.add(layer)

        edge = self._canon_edge(edge)
        k = (edge, layer)
        order = len(edge) - 1


        if k not in self._edge_list:
            e_id = self._next_edge_id
            self._reverse_edge_list[e_id] = k
            self._edge_list[k] = e_id
            self._next_edge_id += 1
            self._weights[e_id] = weight
        elif k in self._edge_list and self._weighted:
            self._weights[self._edge_list[k]] += weight

        e_id = self._edge_list[k]

        if metadata is None:
            metadata = {}

        self._edge_metadata[k] = metadata

        for node in edge:
            self.add_node(node)

        for node in edge:
            if e_id not in self._adj[node]:
                self._adj[node].append(e_id)

    def add_edges(self, edge_list, edge_layer=None, weights=None, metadata=None) -> None:
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

        if edge_layer is None:
            raise ValueError("edge_layer must be provided for MultiplexHypergraph")

        for i, edge in enumerate(edge_list):
            self.add_edge(
                edge,
                edge_layer[i],
                weight=(
                    weights[i] if self._weighted and weights is not None else None
                ),
                metadata=metadata[i] if metadata is not None else None,
            )

    def remove_edge(self, edge, layer=None) -> None:
        """
        Remove an edge from the multiplex hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to remove. Can be either the edge nodes or ((nodes...), layer).
        layer : str, optional
            The layer of the edge. Required if edge is just the nodes.

        Raises
        ------
        ValueError
            If the edge is not in the hypergraph.
        """
        # Handle both formats: edge as (nodes, layer) or separate edge and layer
        if layer is None:
            if isinstance(edge, tuple) and len(edge) == 2 and isinstance(edge[1], str):
                nodes, layer = edge
                edge_key = (self._canon_edge(nodes), layer)
            else:
                raise ValueError("Layer must be provided or edge must be in format ((nodes...), layer)")
        else:
            edge_key = (self._canon_edge(edge), layer)

        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")

        edge_id = self._edge_list[edge_key]

        del self._reverse_edge_list[edge_id]
        if edge_id in self._weights:
            del self._weights[edge_id]
        if edge in self._edge_metadata.keys():
            del self._edge_metadata[edge]

        nodes, layer = edge_key
        for node in nodes:
            if edge_id in self._adj[node]:
                self._adj[node].remove(edge_id)

        del self._edge_list[edge_key]

    def remove_edges(self, edge_list) -> None:
        """
        Remove multiple edges from the hypergraph.

        Parameters
        ----------
        edge_list : list
            The list of edges to remove. Each edge should be in format ((nodes...), layer).

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any edge is not in the hypergraph.
        """
        for edge in edge_list:
            self.remove_edge(edge)

    def get_edges(self, metadata: bool = False, layer: str = None):
        """
        Get edges from the hypergraph.

        Parameters
        ----------
        metadata : bool, optional
            If True, return edge metadata dictionary. If False, return list of edges.
        layer : str, optional
            If provided, only return edges from the specified layer.

        Returns
        -------
        list or dict
            List of edges or dictionary of edge metadata.
        """
        if layer is not None:
            # Filter edges by layer
            filtered_edges = {k: v for k, v in self._edge_list.items() if k[1] == layer}
            if metadata:
                return {
                    self._reverse_edge_list[v]: self._edge_metadata[k]
                    for k, v in filtered_edges.items()
                }
            else:
                return list(filtered_edges.keys())
        else:
            if metadata:
                return {
                    self._reverse_edge_list[v]: self._edge_metadata[k]
                    for k, v in self._edge_list.items()
                }
            else:
                return list(self._edge_list.keys())

    def get_neighbors(self, node, order: int = None, size: int = None):
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
            If order and size are both specified or neither are specified.
        """
        if order is not None and size is not None:
            raise ValueError("Cannot specify both order and size")
        if order is None and size is None:
            target_size = None
        elif order is not None:
            target_size = order + 1
        else:
            target_size = size

        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))

        neighbors = set()
        for edge_id in self._adj[node]:
            edge, layer = self._reverse_edge_list[edge_id]
            if target_size is None or len(edge) == target_size:
                neighbors.update(n for n in edge if n != node)

        return neighbors

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
        if order is not None and size is not None:
            raise ValueError("Cannot specify both order and size")
        if order is None and size is None:
            target_size = None
        elif order is not None:
            target_size = order + 1
        else:
            target_size = size

        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))

        incident_edges = []
        for edge_id in self._adj[node]:
            edge, layer = self._reverse_edge_list[edge_id]
            if target_size is None or len(edge) == target_size:
                incident_edges.append((edge, layer))

        return incident_edges

    # =============================================================================
    # Weight Management (Implementation of abstract methods)
    # =============================================================================

    def get_weight(self, edge, layer=None):
        """Returns the weight of the specified edge.

        Parameters
        ----------
        edge : tuple
            The edge to get the weight of.
        layer : str, optional
            The layer of the edge. Required if edge format doesn't include layer.

        Returns
        -------
        float
            Weight of the specified edge.
        """
        if layer is None:
            if isinstance(edge, tuple) and len(edge) == 2 and isinstance(edge[1], str):
                nodes, layer = edge
                k = (self._canon_edge(nodes), layer)
            else:
                raise ValueError("Layer must be provided or edge must be in format ((nodes...), layer)")
        else:
            k = (self._canon_edge(edge), layer)

        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(k))
        else:
            return self._weights[self._edge_list[k]]

    def set_weight(self, edge, layer, weight) -> None:
        """Sets the weight of the specified edge.

        Parameters
        ----------
        edge : tuple
            The edge to set the weight of.
        weight : float
            The weight to set.
        layer : str, optional
            The layer of the edge. Required if edge format doesn't include layer.

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

        if layer is None:
            if isinstance(edge, tuple) and len(edge) == 2 and isinstance(edge[1], str):
                nodes, layer = edge
                k = (self._canon_edge(nodes), layer)
            else:
                raise ValueError("Layer must be provided or edge must be in format ((nodes...), layer)")
        else:
            k = (self._canon_edge(edge), layer)

        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        else:
            self._weights[self._edge_list[k]] = weight

    def get_weights(self, order=None, size=None, up_to=False, asdict=False, layer=None):
        """Returns the list of weights of the edges in the hypergraph.

        Parameters
        ----------
        order : int, optional
            Order of the edges to get the weights of.
        size : int, optional
            Size of the edges to get the weights of.
        up_to : bool, optional
            If True, it returns the list of weights of the edges of order smaller or equal to the specified order.
        asdict : bool, optional
            If True, return as dictionary mapping edges to weights.
        layer : str, optional
            If provided, only return weights for edges in the specified layer.

        Returns
        -------
        list or dict
            List or dictionary of weights of the edges in the hypergraph.

        Raises
        ------
        ValueError
            If both order and size are specified.
        """
        if order is not None and size is not None:
            raise ValueError("Cannot specify both order and size")
        
        if order is None and size is None:
            target_size = None
        elif order is not None:
            target_size = order + 1
        else:
            target_size = size

        weights = []
        weight_dict = {}

        for edge_key, edge_id in self._edge_list.items():
            nodes, edge_layer = edge_key
            
            # Filter by layer if specified
            if layer is not None and edge_layer != layer:
                continue
                
            edge_size = len(nodes)
            
            # Filter by size/order
            if target_size is not None:
                if up_to and edge_size > target_size:
                    continue
                elif not up_to and edge_size != target_size:
                    continue
            
            weight = self._weights.get(edge_id, 1)
            weights.append(weight)
            if asdict:
                weight_dict[edge_key] = weight

        return weight_dict if asdict else weights

    # =============================================================================
    # Structural Information (Implementation of abstract methods)
    # =============================================================================

    def num_edges(self) -> int:
        """Returns the number of edges in the hypergraph.

        Returns
        -------
        int
            Number of edges in the hypergraph.
        """
        return len(self._edge_list)

    def get_sizes(self) -> List[int]:
        """Returns the list of sizes of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of sizes of the hyperedges in the hypergraph.
        """
        return [len(edge[0]) for edge in self._edge_list.keys()]

    def is_uniform(self) -> bool:
        """
        Check if the hypergraph is uniform, i.e. all hyperedges have the same size.

        Returns
        -------
        bool
            True if the hypergraph is uniform, False otherwise.
        """
        sizes = self.get_sizes()
        return len(set(sizes)) <= 1 if sizes else True

    # =============================================================================
    # Metadata Management (Implementation of abstract methods)
    # =============================================================================

    def get_incidence_metadata(self, edge, node):
        """Get incidence metadata for a specific edge-node pair."""
        # For multiplex hypergraph, we don't have specific incidence metadata implementation
        # Return empty dict for now
        return {}

    def set_incidence_metadata(self, edge, node, metadata):
        """Set incidence metadata for a specific edge-node pair."""
        # For multiplex hypergraph, we don't have specific incidence metadata implementation
        # This could be implemented if needed
        pass

    def get_all_incidences_metadata(self):
        """Get all incidence metadata."""
        return self._incidences_metadata

    def _restructure_query_edge(self, k: Tuple[Tuple, Any], layer=None):
        """
        An implementation-specific helper for modifying a query edge
        prior to metadata retrieval.
        """
        if layer is not None:
            return (k, layer)
        return k

    # =============================================================================
    # Utility Methods (Implementation of abstract methods)
    # =============================================================================

    def _canon_edge(self, edge: Tuple) -> Tuple:
        """
        Gets the canonical form of an edge (sorts the inner tuples)
        """
        edge = tuple(edge)

        if len(edge) == 2:
            if isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
                # Sort the inner tuples and return
                return (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
            elif not isinstance(edge[0], tuple) and not isinstance(edge[1], tuple):
                # Sort the edge itself if it contains IDs (non-tuple elements)
                return tuple(sorted(edge))

        return tuple(sorted(edge))

    def clear(self):
        """Clear all data from the hypergraph."""
        self._edge_list.clear()
        self._adj.clear()
        self._weights.clear()
        self._hypergraph_metadata.clear()
        self._node_metadata.clear()
        self._edge_metadata.clear()
        self._reverse_edge_list.clear()
        self._existing_layers.clear()
        self._next_edge_id = 0

    # =============================================================================
    # Serialization Support (Implementation of abstract methods)
    # =============================================================================

    def expose_data_structures(self) -> Dict:
        """
        Expose the internal data structures of the multiplex hypergraph for serialization.

        Returns
        -------
        dict
            A dictionary containing all internal attributes of the multiplex hypergraph.
        """
        return {
            "type": "MultiplexHypergraph",
            "hypergraph_metadata": self._hypergraph_metadata,
            "node_metadata": self._node_metadata,
            "edge_metadata": self._edge_metadata,
            "_weighted": self._weighted,
            "_weights": self._weights,
            "_edge_list": self._edge_list,
            "_adj": self._adj,
            "reverse_edge_list": self._reverse_edge_list,
            "next_edge_id": self._next_edge_id,
            "existing_layers": self._existing_layers,
            "_incidences_metadata": self._incidences_metadata,
        }

    def populate_from_dict(self, data: Dict) -> None:
        """
        Populate the attributes of the multiplex hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        self._hypergraph_metadata = data.get("hypergraph_metadata", {})
        self._node_metadata = data.get("node_metadata", {})
        self._edge_metadata = data.get("edge_metadata", {})
        self._weighted = data.get("_weighted", False)
        self._weights = data.get("_weights", {})
        self._edge_list = data.get("_edge_list", {})
        self._adj = data.get("_adj", {})
        self._reverse_edge_list = data.get("reverse_edge_list", {})
        self._next_edge_id = data.get("next_edge_id", 0)
        self._existing_layers = data.get("existing_layers", set())
        self._incidences_metadata = data.get("_incidences_metadata", {})

    def expose_attributes_for_hashing(self) -> dict:
        """
        Expose relevant attributes for hashing specific to MultiplexHypergraph.

        Returns
        -------
        dict
            A dictionary containing key attributes.
        """
        edges = []
        for edge in sorted(self._edge_list.keys()):
            edge = (tuple(sorted(edge[0])), edge[1])
            edge_id = self._edge_list[edge]
            edges.append(
                {
                    "nodes": edge,
                    "weight": self._weights.get(edge_id, 1),
                    "metadata": self.get_edge_metadata(edge=edge[0], layer=edge[1]),
                }
            )

        nodes = []
        for node in sorted(self._node_metadata.keys()):
            nodes.append({"node": node, "metadata": self._node_metadata[node]})

        return {
            "type": "MultiplexHypergraph",
            "weighted": self._weighted,
            "hypergraph_metadata": self._hypergraph_metadata,
            "edges": edges,
            "nodes": nodes,
        }

    # =============================================================================
    # Multiplex-specific methods
    # =============================================================================

    def get_adj_dict(self):
        """Get the adjacency dictionary."""
        return self._adj

    def set_adj_dict(self, adj_dict):
        """Set the adjacency dictionary."""
        self._adj = adj_dict

    def get_existing_layers(self):
        """Get the set of existing layers."""
        return self._existing_layers

    def set_existing_layers(self, existing_layers):
        """Set the existing layers."""
        self._existing_layers = existing_layers

    def degree(self, node, order=None, size=None):
        """Get the degree of a node."""
        from hypergraphx.measures.degree import degree
        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        """Get the degree sequence of the hypergraph."""
        from hypergraphx.measures.degree import degree_sequence
        return degree_sequence(self, order=order, size=size)

    def set_dataset_metadata(self, metadata):
        """Set dataset-level metadata."""
        self._hypergraph_metadata["multiplex_metadata"] = metadata

    def get_dataset_metadata(self):
        """Get dataset-level metadata."""
        return self._hypergraph_metadata.get("multiplex_metadata", {})

    def set_layer_metadata(self, layer_name, metadata):
        """Set metadata for a specific layer."""
        if layer_name not in self._hypergraph_metadata:
            self._hypergraph_metadata[layer_name] = {}
        self._hypergraph_metadata[layer_name] = metadata

    def get_layer_metadata(self, layer_name):
        """Get metadata for a specific layer."""
        return self._hypergraph_metadata.get(layer_name, {})

    def aggregated_hypergraph(self):
        """Create an aggregated hypergraph combining all layers."""
        h = Hypergraph(
            weighted=self._weighted, hypergraph_metadata=self._hypergraph_metadata
        )
        for node in self.get_nodes():
            h.add_node(node, metadata=self._node_metadata[node])
        for edge in self.get_edges():
            _edge, layer = edge
            h.add_edge(
                _edge,
                weight=self.get_weight(_edge, layer),
                metadata=self.get_edge_metadata(_edge, layer),
            )
        return h