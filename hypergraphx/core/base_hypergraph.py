import copy

from .serialization import SerializationMixin


class BaseHypergraph(SerializationMixin):
    """
    Shared implementation for hypergraph-like data structures.

    Subclasses are responsible for:
    - initializing adjacency maps before calling _init_base
    - defining edge normalization and node extraction hooks
    - overriding incidence handling where needed (e.g., directed edges)

    Hook contract:
    - _normalize_edge(edge, **kwargs) -> edge_key
    - _edge_nodes(edge_key) -> iterable of nodes
    - _edge_size(edge_key) -> int (uses _edge_nodes by default)
    - _edge_key_without_node(edge_key, node) -> edge_key with node removed
    - _add_edge(edge_key, weight, metadata) for custom incidence behavior
    - _new_like() -> new instance of the same class
    - _hash_edge_nodes(edge_key) -> node representation for hashing/serialization
    """

    _missing_node_exc = ValueError

    def _init_base(self, weighted=False, hypergraph_metadata=None, node_metadata=None):
        self._weighted = weighted
        self._hypergraph_metadata = hypergraph_metadata or {}
        self._node_metadata = {}
        self._edge_metadata = {}
        self._incidences_metadata = {}
        self._edge_list = {}
        self._reverse_edge_list = {}
        self._weights = {}
        self._next_edge_id = 0
        if node_metadata:
            for node, metadata in node_metadata.items():
                self.add_node(node, metadata=metadata)

    # Hooks
    def _adjacency_maps(self):
        return {"default": self._adj}

    def _primary_adj_map(self):
        return next(iter(self._adjacency_maps().values()))

    def _init_node_adjacency(self, node):
        self._primary_adj_map()[node] = []

    def _normalize_edge(self, edge, **kwargs):
        return edge

    def _edge_nodes(self, edge_key):
        return edge_key

    def _edge_size(self, edge_key):
        return len(self._edge_nodes(edge_key))

    def _edge_order(self, edge_key):
        return self._edge_size(edge_key) - 1

    def _edge_key_without_node(self, edge_key, node):
        return tuple(n for n in edge_key if n != node)

    def _allow_empty_edge(self):
        return False

    def _new_like(self):
        raise RuntimeError(
            "BaseHypergraph cannot instantiate subhypergraphs without a subclass override."
        )

    def _raise_missing_node(self, node):
        raise self._missing_node_exc(f"Node {node} not in hypergraph.")

    def _type_name(self):
        return self.__class__.__name__

    def _extra_data_structures(self):
        return {}

    def _populate_extra_data(self, data):
        return None

    def _hash_edge_nodes(self, edge_key):
        return sorted(self._edge_nodes(edge_key))

    # Core node methods
    def add_node(self, node, metadata=None):
        """Add a node to the hypergraph if it does not already exist."""
        if metadata is None:
            metadata = {}
        primary_adj = self._primary_adj_map()
        if node not in primary_adj:
            self._init_node_adjacency(node)
            self._node_metadata[node] = {}
        if self._node_metadata[node] == {}:
            self._node_metadata[node] = metadata

    def add_nodes(self, node_list, metadata=None):
        """Add multiple nodes to the hypergraph."""
        for node in node_list:
            try:
                self.add_node(node, metadata[node] if metadata is not None else None)
            except KeyError:
                raise ValueError(
                    "The metadata dictionary must contain an entry for each node in the node list."
                )

    def remove_node(self, node, keep_edges=False):
        """Remove a node, optionally preserving incident edges without it."""
        primary_adj = self._primary_adj_map()
        if node not in primary_adj:
            self._raise_missing_node(node)

        edge_ids = list(primary_adj[node])
        if keep_edges:
            for edge_id in edge_ids:
                edge_key = self._reverse_edge_list[edge_id]
                updated_key = self._edge_key_without_node(edge_key, node)
                weight = self._weights.get(edge_id, 1)
                metadata = self._edge_metadata.get(edge_id, {})
                self._remove_edge_key(edge_key)
                if self._allow_empty_edge() or self._edge_size(updated_key) > 0:
                    self._add_edge(updated_key, weight=weight, metadata=metadata)
        else:
            for edge_id in edge_ids:
                edge_key = self._reverse_edge_list[edge_id]
                self._remove_edge_key(edge_key)

        del primary_adj[node]
        if node in self._node_metadata:
            del self._node_metadata[node]

    def remove_nodes(self, node_list, keep_edges=False):
        """Remove multiple nodes from the hypergraph."""
        for node in node_list:
            self.remove_node(node, keep_edges=keep_edges)

    def get_nodes(self, metadata=False):
        """Return nodes, optionally with their metadata."""
        if not metadata:
            return list(self._primary_adj_map().keys())
        return {node: self._node_metadata[node] for node in self._primary_adj_map()}

    def check_node(self, node):
        """Return True if the node exists in the hypergraph."""
        return node in self._primary_adj_map()

    def get_incident_edges(self, node, order=None, size=None):
        """Return edges incident to a node, optionally filtered by order or size."""
        primary_adj = self._primary_adj_map()
        if node not in primary_adj:
            self._raise_missing_node(node)
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        edges = [self._reverse_edge_list[edge_id] for edge_id in primary_adj[node]]
        return self._filter_edges_by_order(edges, order=order, size=size)

    def get_neighbors(self, node, order=None, size=None):
        """Return the set of neighbors of a node via incident edges."""
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        edges = self.get_incident_edges(node, order=order, size=size)
        neighbors = set()
        for edge_key in edges:
            neighbors.update(self._edge_nodes(edge_key))
        neighbors.discard(node)
        return neighbors

    # Incidence helpers
    def _add_incidence(self, node, edge_id, edge_key):
        for adj in self._adjacency_maps().values():
            adj[node].append(edge_id)

    def _remove_incidence(self, node, edge_id, edge_key):
        for adj in self._adjacency_maps().values():
            if node in adj and edge_id in adj[node]:
                adj[node].remove(edge_id)

    # Edge helpers
    def _validate_weight(self, weight):
        if not self._weighted and weight is not None and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )
        if weight is None:
            return 1
        return weight

    def _add_edge_key(self, edge_key, weight, metadata):
        if edge_key not in self._edge_list:
            edge_id = self._next_edge_id
            self._next_edge_id += 1
            self._edge_list[edge_key] = edge_id
            self._reverse_edge_list[edge_id] = edge_key
            self._weights[edge_id] = 1 if not self._weighted else weight
        elif self._weighted:
            self._weights[self._edge_list[edge_key]] += weight

        edge_id = self._edge_list[edge_key]
        self._edge_metadata[edge_id] = metadata or {}
        return edge_id

    def _add_edge(self, edge_key, weight=None, metadata=None):
        weight = self._validate_weight(weight)
        edge_id = self._add_edge_key(edge_key, weight=weight, metadata=metadata)
        for node in self._edge_nodes(edge_key):
            self.add_node(node)
            self._add_incidence(node, edge_id, edge_key)

    def _remove_edge_key(self, edge_key):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        edge_id = self._edge_list[edge_key]
        for node in self._edge_nodes(edge_key):
            self._remove_incidence(node, edge_id, edge_key)
        for key in list(self._incidences_metadata):
            if key[0] == edge_key:
                del self._incidences_metadata[key]
        del self._reverse_edge_list[edge_id]
        if edge_id in self._weights:
            del self._weights[edge_id]
        if edge_id in self._edge_metadata:
            del self._edge_metadata[edge_id]
        del self._edge_list[edge_key]

    def _edge_exists(self, edge_key):
        return edge_key in self._edge_list

    # Query helpers
    def _filter_edges_by_order(self, edges, order=None, size=None, up_to=False):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return list(edges)
        if size is not None:
            order = size - 1
        if not up_to:
            return [edge for edge in edges if self._edge_order(edge) == order]
        return [edge for edge in edges if self._edge_order(edge) <= order]

    def _get_edges_common(
        self,
        order=None,
        size=None,
        up_to=False,
        subhypergraph=False,
        keep_isolated_nodes=False,
        metadata=False,
    ):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if not subhypergraph and keep_isolated_nodes:
            raise ValueError("Cannot keep nodes if not returning subhypergraphs.")

        edges = self._filter_edges_by_order(
            self._edge_list.keys(), order=order, size=size, up_to=up_to
        )

        if subhypergraph:
            h = self._new_like()
            if keep_isolated_nodes:
                h.add_nodes(list(self.get_nodes()))
                for node in h.get_nodes():
                    h.set_node_metadata(node, self.get_node_metadata(node))
            if self._weighted:
                edge_weights = [self.get_weight(edge) for edge in edges]
                h.add_edges(edges, edge_weights)
            else:
                h.add_edges(edges)
            for edge in edges:
                h.set_edge_metadata(edge, self.get_edge_metadata(edge))
            return h

        if metadata:
            return {edge: self.get_edge_metadata(edge) for edge in edges}
        return edges

    # Weights
    def set_weight(self, edge_key, weight):
        weight = self._validate_weight(weight)
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        edge_id = self._edge_list[edge_key]
        self._weights[edge_id] = weight

    def get_weight(self, edge_key):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        edge_id = self._edge_list[edge_key]
        return self._weights[edge_id]

    def get_weights(self, order=None, size=None, up_to=False, asdict=False):
        """Return edge weights, optionally filtered by order or size."""
        w = None
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            w = {
                edge: self._weights[self._edge_list[edge]] for edge in self._edge_list
            }
        if size is not None:
            order = size - 1
        if w is None:
            w = {
                edge: self._weights[self._edge_list[edge]]
                for edge in self._filter_edges_by_order(
                    self._edge_list.keys(), order=order, up_to=up_to
                )
            }
        return w if asdict else list(w.values())

    # Info
    def max_order(self):
        return self.max_size() - 1

    def max_size(self):
        return max(self.get_sizes())

    def num_nodes(self):
        return len(list(self.get_nodes()))

    def num_edges(self, order=None, size=None, up_to=False):
        """Return the number of edges, optionally filtered by order or size."""
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return len(self._edge_list)
        return len(
            self._filter_edges_by_order(
                self._edge_list.keys(), order=order, size=size, up_to=up_to
            )
        )

    def get_sizes(self):
        """Return the size (cardinality) of each edge."""
        return [self._edge_size(edge) for edge in self._edge_list.keys()]

    def distribution_sizes(self):
        from collections import Counter

        return dict(Counter(self.get_sizes()))

    def get_orders(self):
        """Return the order of each edge."""
        return [self._edge_order(edge) for edge in self._edge_list.keys()]

    def is_weighted(self):
        return self._weighted

    def is_uniform(self):
        """Return True if all edges have the same size."""
        uniform = True
        sz = None
        for edge in self._edge_list:
            edge_size = self._edge_size(edge)
            if sz is None:
                sz = edge_size
            elif edge_size != sz:
                uniform = False
                break
        return uniform

    # Metadata
    def set_hypergraph_metadata(self, metadata):
        self._hypergraph_metadata = metadata

    def get_hypergraph_metadata(self):
        return self._hypergraph_metadata

    def set_node_metadata(self, node, metadata):
        if node not in self._node_metadata:
            self._raise_missing_node(node)
        self._node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._node_metadata:
            self._raise_missing_node(node)
        return self._node_metadata[node]

    def get_all_nodes_metadata(self):
        return self._node_metadata

    def set_edge_metadata(self, edge_key, metadata):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        self._edge_metadata[self._edge_list[edge_key]] = metadata

    def get_edge_metadata(self, edge_key):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        return self._edge_metadata[self._edge_list[edge_key]]

    def get_all_edges_metadata(self):
        return self._edge_metadata

    def set_incidence_metadata(self, edge_key, node, metadata):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        self._incidences_metadata[(edge_key, node)] = metadata

    def get_incidence_metadata(self, edge_key, node):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        return self._incidences_metadata[(edge_key, node)]

    def get_all_incidences_metadata(self):
        return {k: v for k, v in self._incidences_metadata.items()}

    def set_attr_to_hypergraph_metadata(self, field, value):
        self._hypergraph_metadata[field] = value

    def set_attr_to_node_metadata(self, node, field, value):
        if node not in self._node_metadata:
            self._raise_missing_node(node)
        self._node_metadata[node][field] = value

    def set_attr_to_edge_metadata(self, edge_key, field, value):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        self._edge_metadata[self._edge_list[edge_key]][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self._node_metadata:
            self._raise_missing_node(node)
        del self._node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge_key, field):
        if edge_key not in self._edge_list:
            raise ValueError(f"Edge {edge_key} not in hypergraph.")
        del self._edge_metadata[self._edge_list[edge_key]][field]

    # Basic functions
    def clear(self):
        self._edge_list.clear()
        for adj in self._adjacency_maps().values():
            adj.clear()
        self._weights.clear()
        self._hypergraph_metadata.clear()
        self._incidences_metadata.clear()
        self._node_metadata.clear()
        self._edge_metadata.clear()
        self._reverse_edge_list.clear()

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        title = "Hypergraph with {} nodes and {} edges.\n".format(
            self.num_nodes(), self.num_edges()
        )
        details = "Distribution of hyperedge sizes: {}".format(
            self.distribution_sizes()
        )
        return title + details

    def __len__(self):
        return len(self._edge_list)

    def __iter__(self):
        return iter(self._edge_list.items())

    def _expose_adjacency_data(self):
        adj_maps = self._adjacency_maps()
        if list(adj_maps.keys()) == ["default"]:
            return {"_adj": adj_maps["default"]}
        return {f"_adj_{name}": adj for name, adj in adj_maps.items()}

    def _populate_adjacency_data(self, data):
        adj_maps = self._adjacency_maps()
        if "default" in adj_maps and "_adj" in data:
            self._adj = data.get("_adj", {})
            return
        for name in adj_maps:
            key = f"_adj_{name}"
            if key in data:
                setattr(self, key, data.get(key, {}))
