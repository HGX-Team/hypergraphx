from hypergraphx import Hypergraph
from hypergraphx.core.i_undirected_hypergraph import IUndirectedHypergraph

class MultiplexHypergraph(IUndirectedHypergraph):
    """
    A Multiplex Hypergraph is a hypergraph where hyperedges are organized into multiple layers.
    Each layer share the same node-set and represents a specific context or relationship between nodes, and hyperedges can
    have weights and metadata specific to their layer.
    """

    def __init__(
        self,
        edge_list=None,
        edge_layer=None,
        weighted=False,
        weights=None,
        hypergraph_metadata=None,
        node_metadata=None,
        edge_metadata=None,
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
            edge_list=edge_list,
            weighted=weighted,
            weights=weights,
            hypergraph_metadata=hypergraph_metadata,
            node_metadata=node_metadata,
            edge_metadata=edge_metadata
        )

        # Configure additional hypergraph metadata
        self._hypergraph_metadata["type"] = "MultiplexHypergraph"
        
        # Initialize other attributes
        self._existing_layers = set()

        # Add node metadata if provided
        if node_metadata:
            for node, metadata in node_metadata.items():
                self.add_node(node, metadata=metadata)

        # Handle edge and layer consistency
        if edge_list is not None and edge_layer is None:
            # Extract layers from edge_list if layer information is embedded
            pairwise_edge_mask = (isinstance(edge, tuple) and len(edge) == 2 for edge in edge_list)
            if all(pairwise_edge_mask):
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


    def get_incident_edges(self, node):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return [self._reverse_edge_list[e_id] for e_id in self._adj[node]]

    def _restructure_query_edge(self, edge, layer):
        # modify the edge being queried by making a tuple of it and the layer
        return (edge, layer)

    def get_existing_layers(self):
        return self._existing_layers

    def set_existing_layers(self, existing_layers):
        self._existing_layers = existing_layers

    def add_edges(self, edge_list, edge_layer, weights=None, metadata=None):
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

        i = 0
        if edge_list is not None:
            for edge in edge_list:
                self.add_edge(
                    edge,
                    edge_layer[i],
                    weight=(
                        weights[i] if self._weighted and weights is not None else None
                    ),
                    metadata=metadata[i] if metadata is not None else None,
                )
                i += 1

    def add_edge(self, edge, layer, weight=None, metadata=None):
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

        self._edge_metadata[e_id] = metadata

        for node in edge:
            self.add_node(node)

        for node in edge:
            self._adj[node].append(e_id)

    def remove_edge(self, edge):
        """
        Remove an edge from the multiplex hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to remove. Should be of the form ((nodes...), layer).

        Raises
        ------
        ValueError
            If the edge is not in the hypergraph.
        """
        edge = self._canon_edge(edge)
        if not self.check_edge(edge):
            raise ValueError(f"Edge {edge} not in hypergraph.")

        edge_id = self._edge_list[edge]

        del self._reverse_edge_list[edge_id]
        if edge_id in self._weights:
            del self._weights[edge_id]
        if edge_id in self._edge_metadata:
            del self._edge_metadata[edge_id]

        nodes, layer = edge
        for node in nodes:
            if edge_id in self._adj[node]:
                self._adj[node].remove(edge_id)

        del self._edge_list[edge]

    def remove_node(self, node, keep_edges=False):
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

                self.remove_edge((edge, layer))
                if updated_edge:
                    self.add_edge(
                        updated_edge,
                        layer,
                        weight=self._weights.get(edge_id, 1),
                        metadata=self._edge_metadata.get(edge_id, {}),
                    )
        else:
            for edge_id in edges_to_process:
                edge, layer = self._reverse_edge_list[edge_id]
                self.remove_edge((edge, layer))

        del self._adj[node]
        if node in self._node_metadata:
            del self._node_metadata[node]

    def get_edges(self, metadata=False):
        if metadata:
            return {
                self._reverse_edge_list[k]: self._edge_metadata[k]
                for k in self._edge_metadata.keys()
            }
        else:
            return list(self._edge_list.keys())

    def set_dataset_metadata(self, metadata):
        self._hypergraph_metadata["multiplex_metadata"] = metadata

    def get_dataset_metadata(self):
        return self._hypergraph_metadata["multiplex_metadata"]

    def set_layer_metadata(self, layer_name, metadata):
        if layer_name not in self._hypergraph_metadata:
            self._hypergraph_metadata[layer_name] = {}
        self._hypergraph_metadata[layer_name] = metadata

    def get_layer_metadata(self, layer_name):
        return self._hypergraph_metadata[layer_name]

    def aggregated_hypergraph(self):
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

    def set_attr_to_edge_metadata(self, edge, layer, field, value):
        edge = self._canon_edge(edge)
        if edge not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._edge_metadata[self._edge_list[(edge, layer)]][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        del self._node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge, layer, field):
        edge = self._canon_edge(edge)
        if edge not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        del self._edge_metadata[self._edge_list[(edge, layer)]][field]

    def clear(self):
        self._adj.clear()
        self._weights.clear()
        self._hypergraph_metadata.clear()
        self._incidences_metadata.clear()
        self._node_metadata.clear()
        self._edge_metadata.clear()
        self._edge_list.clear()
        self._reverse_edge_list.clear()
        self._existing_layers.clear()


    def expose_data_structures(self):
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
        }

    def populate_from_dict(self, data):
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
                    "metadata": self._edge_metadata.get(edge_id, {}),
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
