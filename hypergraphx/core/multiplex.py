from .undirected import Hypergraph
from hypergraphx.core.base import BaseHypergraph
from hypergraphx.utils.edges import canon_edge


class MultiplexHypergraph(BaseHypergraph):
    """
    A Multiplex Hypergraph is a hypergraph where hyperedges are organized into multiple layers.
    Each layer share the same node-set and represents a specific context or relationship between nodes, and hyperedges can
    have weights and metadata specific to their layer.
    """

    def __init__(
        self,
        edge_list=None,
        edge_layer=None,
        weighted=True,
        weights=None,
        hypergraph_metadata=None,
        node_metadata=None,
        edge_metadata=None,
        duplicate_policy=None,
        metadata_policy=None,
    ):
        """
        Initialize a Multiplex Hypergraph with optional edges, layers, weights, and metadata.

        Parameters
        ----------
        edge_list : list of tuples, optional
            A list of edges where each edge is represented as a tuple of nodes.
            If `edge_layer` is not provided, each tuple in `edge_list` should have
            the format `(layer, edge)`, where `edge` is itself a tuple of nodes.
        edge_layer : list of str, optional
            A list of layer names corresponding to each edge in `edge_list`.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is True.
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
        self._adj = {}
        self._existing_layers = set()
        metadata = hypergraph_metadata or {}
        metadata.update({"weighted": weighted, "type": "MultiplexHypergraph"})
        self._init_base(
            weighted=weighted,
            hypergraph_metadata=metadata,
            node_metadata=node_metadata,
            duplicate_policy=duplicate_policy,
            metadata_policy=metadata_policy,
        )

        # Handle edge and layer consistency
        if edge_list is not None and edge_layer is None:
            # Extract layers from edge_list if layer information is embedded
            if all(
                isinstance(e, tuple)
                and len(e) == 2
                and isinstance(e[0], str)
                and isinstance(e[1], (tuple, list))
                for e in edge_list
            ):
                edge_layer = [e[0] for e in edge_list]
                edge_list = [e[1] for e in edge_list]
            elif all(
                isinstance(e, tuple)
                and len(e) == 2
                and isinstance(e[0], (tuple, list))
                and isinstance(e[1], str)
                for e in edge_list
            ):
                # Backward compatible: (edge, layer)
                edge_layer = [e[1] for e in edge_list]
                edge_list = [e[0] for e in edge_list]
            else:
                raise ValueError(
                    "If edge_layer is not provided, edge_list must contain tuples of the form (layer, edge)."
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

    def _normalize_edge(self, edge, layer=None, **kwargs):
        """
        Normalize multiplex edges to a canonical edge key: (layer, canon_edge(edge)).

        Public API convention: multiplex edges are represented as an edge key
        `(layer, edge)` at boundaries. Most methods accept either:
        - separate arguments: `edge=<tuple>, layer=<str>`
        - a packed edge key: `edge=(<tuple>, <str>)` and `layer=None`

        Note: to avoid ambiguity with 2-node hyperedges, a packed edge key is
        only inferred when the first element looks like a layer (string).
        """
        if layer is None:
            if isinstance(edge, tuple) and len(edge) == 2 and isinstance(edge[0], str):
                layer, edge = edge
            elif (
                isinstance(edge, tuple)
                and len(edge) == 2
                and isinstance(edge[1], str)
                and isinstance(edge[0], (tuple, list))
            ):
                # Backward compatible: packed (edge, layer)
                edge, layer = edge
            else:
                raise ValueError(
                    "Multiplex edges must include a layer: pass `layer=...` or a packed `(layer, edge)` tuple."
                )
        return (layer, canon_edge(edge))

    def _edge_nodes(self, edge_key):
        return edge_key[1]

    def _edge_key_without_node(self, edge_key, node):
        layer, edge = edge_key
        return (layer, tuple(n for n in edge if n != node))

    def _allow_empty_edge(self):
        return False

    def _new_like(self):
        return MultiplexHypergraph(weighted=self._weighted)

    def _hash_edge_nodes(self, edge_key):
        layer, edge = edge_key
        return (tuple(sorted(edge)), layer)

    def _extra_data_structures(self):
        return {"existing_layers": self._existing_layers}

    def _populate_extra_data(self, data):
        self._existing_layers = data.get("existing_layers", set())

    def get_adj_dict(self):
        return self._adj

    def set_adj_dict(self, adj_dict):
        self._guard_unsafe_setter("MultiplexHypergraph.set_adj_dict")
        self._adj = adj_dict
        self._maybe_validate_invariants()

    def get_incident_edges(self, node):
        return super().get_incident_edges(node)

    def degree(self, node, order=None, size=None):
        from hypergraphx.measures.degree import degree

        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_sequence

        return degree_sequence(self, order=order, size=size)

    def get_edge_metadata(self, edge, layer=None):
        edge_key = self._normalize_edge(edge, layer=layer)
        return super().get_edge_metadata(edge_key)

    def is_weighted(self):
        return self._weighted

    def get_edge_list(self):
        return self._edge_list

    def set_edge_list(self, edge_list):
        self._guard_unsafe_setter("MultiplexHypergraph.set_edge_list")
        self._edge_list = edge_list
        self._maybe_validate_invariants()

    def get_existing_layers(self):
        return self._existing_layers

    def set_existing_layers(self, existing_layers):
        self._existing_layers = existing_layers

    def get_nodes(self, metadata=False):
        return super().get_nodes(metadata=metadata)

    def add_node(self, node, metadata=None):
        """
        Add a node to the hypergraph. If the node is already in the hypergraph, nothing happens.

        Parameters
        ----------
        node : object
            The node to add.

        Returns
        -------
        None
        """
        super().add_node(node, metadata=metadata)

    def add_nodes(self, node_list: list, node_metadata=None):
        """
        Add a list of nodes to the hypergraph.

        Parameters
        ----------
        node_list : list
            The list of nodes to add.

        Returns
        -------
        None
        """
        super().add_nodes(node_list, metadata=node_metadata)

    def add_edges(
        self,
        edge_list,
        edge_layer=None,
        weights=None,
        metadata=None,
    ):
        """Add a list of hyperedges to the hypergraph. If a hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge_list : list
            The list of hyperedges to add.

        edge_layer : list, optional
            The list of layers to which the hyperedges belong. If not provided,
            `edge_list` must contain packed `(edge, layer)` tuples.

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
        if edge_list is not None:
            edge_list = list(edge_list)
        if edge_layer is None and edge_list is not None:
            if all(
                isinstance(e, tuple)
                and len(e) == 2
                and isinstance(e[0], str)
                and isinstance(e[1], (tuple, list))
                for e in edge_list
            ):
                edge_layer = [e[0] for e in edge_list]
                edge_list = [e[1] for e in edge_list]
            elif all(
                isinstance(e, tuple)
                and len(e) == 2
                and isinstance(e[0], (tuple, list))
                and isinstance(e[1], str)
                for e in edge_list
            ):
                # Backward compatible: packed (edge, layer)
                edge_layer = [e[1] for e in edge_list]
                edge_list = [e[0] for e in edge_list]
            else:
                raise ValueError(
                    "If edge_layer is not provided, edge_list must contain tuples of the form (layer, edge)."
                )
        if edge_layer is not None:
            edge_layer = list(edge_layer)
        if weights is not None:
            weights = list(weights)
        if metadata is not None:
            metadata = list(metadata)

        if weights is not None:
            if len(edge_list) != len(weights):
                raise ValueError("The number of edges and weights must be the same.")
            if not self._weighted:
                for weight in weights:
                    if weight not in (None, 1):
                        raise ValueError(
                            "If the hypergraph is not weighted, weight can be 1 or None."
                        )

        if edge_list is not None:
            for i, edge in enumerate(edge_list):
                self.add_edge(
                    edge,
                    edge_layer[i],
                    weight=(
                        weights[i] if self._weighted and weights is not None else None
                    ),
                    metadata=metadata[i] if metadata is not None else None,
                )

    def add_edge(
        self,
        edge,
        layer=None,
        weight=None,
        metadata=None,
    ):
        """Add a hyperedge to the hypergraph. If the hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge : tuple
            The hyperedge to add.
        layer : str, optional
            The layer to which the hyperedge belongs. If not provided, `edge`
            must be a packed `(edge, layer)` tuple.
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
        Notes
        -----
        No multi-edges: duplicates never create a new edge. Control behavior via:
        - duplicate_policy: 'error' | 'ignore' | 'accumulate_weight' | 'replace_weight'
        - metadata_policy: 'replace' | 'merge' | 'ignore'

        Incidence metadata is not modified by duplicate adds; use incidence-metadata APIs explicitly.
        """
        edge_key = self._normalize_edge(edge, layer=layer)
        self._existing_layers.add(edge_key[0])
        self._add_edge(edge_key, weight=weight, metadata=metadata)

    def remove_edge(self, edge, layer=None):
        """
        Remove an edge from the multiplex hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to remove. Can be passed as:
            - `edge=(nodes...)` with `layer=<str>`
            - packed `(layer, edge)` tuple with `layer=None`

        Raises
        ------
        ValueError
            If the edge is not in the hypergraph.
        """
        edge_key = self._normalize_edge(edge, layer=layer)
        self._remove_edge_key(edge_key)

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
        super().remove_node(node, keep_edges=keep_edges)

    def get_edges(
        self,
        *,
        layer: str | None = None,
        order: int | None = None,
        size: int | None = None,
        up_to: bool = False,
        metadata: bool = False,
    ):
        """
        Get multiplex edges (edge keys).

        Parameters
        ----------
        layer : str, optional
            If provided, return only edges in this layer.
        order : int, optional
            Edge order filter (order = size - 1). Mutually exclusive with `size`.
        size : int, optional
            Edge size filter (cardinality). Mutually exclusive with `order`.
        up_to : bool, optional
            If True, include edges with order <= `order` (or size <= `size`).
        metadata : bool, optional
            If True, return a dict mapping edge key -> edge metadata.

        Returns
        -------
        list | dict
            List of edge keys `(layer, edge)` or a dict `{edge_key: metadata}`.
        """
        if order is not None and size is not None:
            raise InvalidParameterError("Order and size cannot be both specified.")

        edges = list(self._edge_list.keys())
        if layer is not None:
            edges = [
                e
                for e in edges
                if isinstance(e, tuple) and len(e) == 2 and e[0] == layer
            ]

        edges = self._filter_edges_by_order(edges, order=order, size=size, up_to=up_to)

        if metadata:
            return {edge: self.get_edge_metadata(edge) for edge in edges}
        return edges

    def get_weight(self, edge, layer=None):
        edge_key = self._normalize_edge(edge, layer=layer)
        return super().get_weight(edge_key)

    def set_weight(self, edge, layer=None, weight=None):
        """
        Set edge weight.

        Accepts:
        - `set_weight(edge, layer, weight)` (legacy, explicit layer)
        - `set_weight((layer, edge), weight=<...>)` (packed edge key)
        - `set_weight((layer, edge), <weight>)` (packed edge key, positional weight)
        - `set_weight((edge, layer), <weight>)` (legacy packed edge key, positional weight)
        """
        if weight is None:
            if layer is None:
                raise TypeError("set_weight() missing required argument: 'weight'")
            # Support set_weight((edge, layer), weight) with positional weight
            if (
                isinstance(edge, tuple)
                and len(edge) == 2
                and not isinstance(layer, str)
                and (
                    (isinstance(edge[0], str) and isinstance(edge[1], (tuple, list)))
                    or (isinstance(edge[1], str) and isinstance(edge[0], (tuple, list)))
                )
            ):
                weight = layer
                layer = None
            else:
                raise TypeError("set_weight() missing required argument: 'weight'")
        edge_key = self._normalize_edge(edge, layer=layer)
        super().set_weight(edge_key, weight)

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

    def get_hypergraph_metadata(self):
        return self._hypergraph_metadata

    def set_hypergraph_metadata(self, metadata):
        self._hypergraph_metadata = metadata

    def aggregated_hypergraph(self):
        return self.to_hypergraph()

    def to_hypergraph(
        self,
        keep_node_metadata: bool = True,
        keep_edge_metadata: bool = True,
        keep_hypergraph_metadata: bool = True,
    ):
        """Convert to an undirected Hypergraph by dropping layer information.

        Duplicate hyperedges are merged by summing weights and merging metadata.
        """
        from hypergraphx.utils.metadata import merge_metadata

        hg = Hypergraph(weighted=True)
        if keep_hypergraph_metadata:
            meta = merge_metadata(
                self.get_hypergraph_metadata(),
                {"converted_from": "MultiplexHypergraph"},
            )
            hg.set_hypergraph_metadata(meta)

        if keep_node_metadata:
            for node, metadata in self.get_all_nodes_metadata().items():
                hg.add_node(node, metadata=metadata)

        edge_weights = {}
        edge_metadata = {}
        for layer, edge in self.get_edges():
            edge_weights[edge] = edge_weights.get(edge, 0) + self.get_weight(
                edge, layer
            )
            if keep_edge_metadata:
                edge_metadata[edge] = merge_metadata(
                    edge_metadata.get(edge), self.get_edge_metadata(edge, layer)
                )

        for edge, weight in edge_weights.items():
            hg.add_edge(edge, weight=weight, metadata=edge_metadata.get(edge))

        return hg

    def set_attr_to_hypergraph_metadata(self, field, value):
        self._hypergraph_metadata[field] = value

    def set_attr_to_node_metadata(self, node, field, value):
        super().set_attr_to_node_metadata(node, field, value)

    def set_attr_to_edge_metadata(self, edge, layer, field, value):
        edge_key = self._normalize_edge(edge, layer=layer)
        super().set_attr_to_edge_metadata(edge_key, field, value)

    def remove_attr_from_node_metadata(self, node, field):
        super().remove_attr_from_node_metadata(node, field)

    def remove_attr_from_edge_metadata(self, edge, layer, field):
        edge_key = self._normalize_edge(edge, layer=layer)
        super().remove_attr_from_edge_metadata(edge_key, field)

    # Edge-key convenience for metadata attribute helpers
    def set_attr_to_edge_metadata_key(self, edge_key, field, value):
        edge_key = self._normalize_edge(edge_key, layer=None)
        super().set_attr_to_edge_metadata(edge_key, field, value)

    def remove_attr_from_edge_metadata_key(self, edge_key, field):
        edge_key = self._normalize_edge(edge_key, layer=None)
        super().remove_attr_from_edge_metadata(edge_key, field)

    def __repr__(self):
        return "{}(nodes={}, edges={}, layers={}, weighted={})".format(
            self._type_name(),
            self.num_nodes(),
            self.num_edges(),
            len(self._existing_layers),
            self._weighted,
        )

    def summary(
        self, *, include_size_distribution: bool = True, max_size_bins: int = 20
    ):
        base = super().summary(
            include_size_distribution=include_size_distribution,
            max_size_bins=max_size_bins,
        )
        base["layers"] = sorted(self._existing_layers)
        base["num_layers"] = len(self._existing_layers)
        return base

    def populate_from_dict(self, data):
        """
        Populate the attributes of the multiplex hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        super().populate_from_dict(data)
