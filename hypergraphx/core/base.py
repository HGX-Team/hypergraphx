import copy
import os
import warnings

from hypergraphx.exceptions import (
    InvalidParameterError,
    MissingEdgeError,
    MissingNodeError,
)
from hypergraphx.utils.metadata import merge_metadata


class SerializationMixin:
    """
    Serialization and hashing helpers for hypergraph-like classes.
    """

    def expose_data_structures(self):
        data = {
            "type": self._type_name(),
            "_weighted": self._weighted,
            "_edge_list": self._edge_list,
            "_weights": self._weights,
            "hypergraph_metadata": self._hypergraph_metadata,
            "node_metadata": self._node_metadata,
            "edge_metadata": self._edge_metadata,
            "incidences_metadata": self._incidences_metadata,
            "reverse_edge_list": self._reverse_edge_list,
            "next_edge_id": self._next_edge_id,
        }
        if hasattr(self, "_empty_edges"):
            data["empty_edges"] = self._empty_edges
        data.update(self._expose_adjacency_data())
        data.update(self._extra_data_structures())
        return data

    def populate_from_dict(self, data):
        self._weighted = data.get("_weighted", False)
        self._edge_list = data.get("_edge_list", {}) or {}
        self._weights = data.get("_weights", {}) or {}
        self._hypergraph_metadata = data.get("hypergraph_metadata", {}) or {}
        self._node_metadata = data.get("node_metadata", {}) or {}
        self._edge_metadata = data.get("edge_metadata", {}) or {}
        self._reverse_edge_list = data.get("reverse_edge_list", {}) or {}
        self._next_edge_id = data.get("next_edge_id", 0)
        self._incidences_metadata = data.get("incidences_metadata", {}) or {}
        if hasattr(self, "_empty_edges"):
            self._empty_edges = data.get("empty_edges", {})
        self._populate_adjacency_data(data)
        self._populate_extra_data(data)
        # If the implementation provides invariant validation, run it optionally.
        maybe_validate = getattr(self, "_maybe_validate_invariants", None)
        if callable(maybe_validate):
            maybe_validate()

    def expose_attributes_for_hashing(self):
        edges = []
        for edge_key in sorted(self._edge_list.keys()):
            edge_id = self._edge_list[edge_key]
            edges.append(
                {
                    "nodes": self._hash_edge_nodes(edge_key),
                    "weight": self._weights.get(edge_id, 1),
                    "metadata": self._edge_metadata.get(edge_id, {}),
                }
            )

        nodes = []
        for node in sorted(self._node_metadata.keys()):
            nodes.append({"node": node, "metadata": self._node_metadata[node]})

        return {
            "type": self._type_name(),
            "weighted": self._weighted,
            "hypergraph_metadata": self._hypergraph_metadata,
            "edges": edges,
            "nodes": nodes,
        }

    def get_mapping(self):
        from hypergraphx.utils.labeling import LabelEncoder

        return LabelEncoder().fit(self.get_nodes())


class BaseHypergraph(SerializationMixin):
    """
    Shared implementation for hypergraph-like data structures.

    Subclasses are responsible for:
    - initializing adjacency maps before calling _init_base
    - defining edge normalization and node extraction hooks
    - overriding incidence handling where needed (e.g., directed edges)

    Hook contract:
    - _normalize_edge(edge, ``**kwargs``) -> edge_key
    - _edge_nodes(edge_key) -> iterable of nodes
    - _edge_size(edge_key) -> int (uses _edge_nodes by default)
    - _edge_key_without_node(edge_key, node) -> edge_key with node removed
    - _add_edge(edge_key, weight, metadata) for custom incidence behavior
    - _new_like() -> new instance of the same class
    - _hash_edge_nodes(edge_key) -> node representation for hashing/serialization

    Duplicate edges / multi-edges:
    - Core hypergraphs do *not* support multi-edges: adding the same edge key twice never creates a new edge.
    - Duplicate handling is controlled via `duplicate_policy` and `metadata_policy` (per instance defaults, overridable per call).
    - If you need multiple distinct “instances” of the same interaction, model them explicitly:
      use a TemporalHypergraph (different `time`) or MultiplexHypergraph (different `layer`).
    """

    _missing_node_exc = MissingNodeError

    def _init_base(
        self,
        weighted=True,
        hypergraph_metadata=None,
        node_metadata=None,
        duplicate_policy=None,
        metadata_policy=None,
    ):
        self._weighted = weighted
        self._hypergraph_metadata = hypergraph_metadata or {}
        self._node_metadata = {}
        self._edge_metadata = {}
        self._incidences_metadata = {}
        self._edge_list = {}
        self._reverse_edge_list = {}
        self._weights = {}
        self._next_edge_id = 0
        # Duplicate-edge policies (no multi-edges: duplicates never create new edges).
        # Defaults:
        # - Unweighted: ignore duplicates
        # - Weighted: accumulate weights on duplicates
        self._duplicate_policy = (
            ("accumulate_weight" if weighted else "ignore")
            if duplicate_policy is None
            else duplicate_policy
        )
        # Metadata default: merge on duplicates to avoid silent overwrite surprises.
        self._metadata_policy = "merge" if metadata_policy is None else metadata_policy
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

    def _validate_metadata_dict(self, metadata, label):
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            raise InvalidParameterError(f"{label} metadata must be a dict.")

    def set_duplicate_policy(self, policy: str) -> None:
        self._duplicate_policy = policy

    def get_duplicate_policy(self) -> str:
        return self._duplicate_policy

    def set_metadata_policy(self, policy: str) -> None:
        self._metadata_policy = policy

    def get_metadata_policy(self) -> str:
        return self._metadata_policy

    @property
    def nodes(self):
        from hypergraphx.core.views import NodeView

        return NodeView(self)

    @property
    def edges(self):
        from hypergraphx.core.views import EdgeView

        return EdgeView(self)

    # Invariants / debug helpers
    def _debug_invariants_enabled(self) -> bool:
        """
        Invariant checks can be expensive. Enable them explicitly via:
        - running Python without -O (i.e. __debug__ is True) AND
        - setting HGX_DEBUG_INVARIANTS=1/true/yes/on.
        """
        if not __debug__:
            return False
        val = os.getenv("HGX_DEBUG_INVARIANTS", "")
        return val.strip().lower() in {"1", "true", "yes", "on"}

    def validate_invariants(self) -> None:
        """Public hook to validate internal consistency (useful in debugging)."""
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """
        Validate internal data-structure consistency.

        This is intended for debugging and test/dev environments.
        """
        # Edge id <-> edge key bijection.
        if len(self._edge_list) != len(self._reverse_edge_list):
            raise RuntimeError(
                "Invariant violated: edge_list and reverse_edge_list size mismatch."
            )
        for edge_key, edge_id in self._edge_list.items():
            if self._reverse_edge_list.get(edge_id) != edge_key:
                raise RuntimeError(
                    "Invariant violated: edge_id <-> edge_key mapping is not a bijection."
                )

        valid_edge_ids = set(self._reverse_edge_list.keys())

        # Weights and edge metadata must align to existing edge_ids.
        for edge_id in self._weights.keys():
            if edge_id not in valid_edge_ids:
                raise RuntimeError(
                    "Invariant violated: weights contain unknown edge_id."
                )
        for edge_id in self._edge_metadata.keys():
            if edge_id not in valid_edge_ids:
                raise RuntimeError(
                    "Invariant violated: edge_metadata contain unknown edge_id."
                )

        # Adjacency lists contain only valid edge_ids and only reference known nodes.
        for name, adj in self._adjacency_maps().items():
            for node, edge_ids in adj.items():
                if node not in self._node_metadata:
                    raise RuntimeError(
                        f"Invariant violated: adjacency map {name!r} references unknown node."
                    )
                for edge_id in edge_ids:
                    if edge_id not in valid_edge_ids:
                        raise RuntimeError(
                            f"Invariant violated: adjacency map {name!r} contains unknown edge_id."
                        )

        # Incidence metadata should refer to existing edges/nodes.
        for (edge_key, node), meta in self._incidences_metadata.items():
            if edge_key not in self._edge_list:
                raise RuntimeError(
                    "Invariant violated: incidence metadata references unknown edge_key."
                )
            if node not in self._node_metadata:
                raise RuntimeError(
                    "Invariant violated: incidence metadata references unknown node."
                )
            if meta is not None and not isinstance(meta, dict):
                raise RuntimeError(
                    "Invariant violated: incidence metadata must be a dict."
                )

    def _maybe_validate_invariants(self) -> None:
        if self._debug_invariants_enabled():
            self._validate_invariants()

    def _allow_unsafe_setters(self) -> bool:
        val = os.getenv("HGX_ALLOW_UNSAFE_SETTERS", "")
        return val.strip().lower() in {"1", "true", "yes", "on"}

    def _guard_unsafe_setter(self, name: str) -> None:
        """
        Guard "invariant grenade" setters that can put the object into an inconsistent state.

        By default they are blocked. Set HGX_ALLOW_UNSAFE_SETTERS=1 to bypass.
        """
        if not self._allow_unsafe_setters():
            raise InvalidParameterError(
                f"{name} is an unsafe operation and is disabled by default. "
                "Use populate_from_dict()/load_* APIs instead, or set "
                "HGX_ALLOW_UNSAFE_SETTERS=1 if you really know what you are doing."
            )
        warnings.warn(
            f"{name} is unsafe and deprecated; it may be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )

    # Core node methods
    def add_node(self, node, metadata=None):
        """Add a node to the hypergraph if it does not already exist."""
        if metadata is None:
            metadata = {}
        self._validate_metadata_dict(metadata, "node")
        primary_adj = self._primary_adj_map()
        if node not in primary_adj:
            self._init_node_adjacency(node)
            self._node_metadata[node] = {}
        if self._node_metadata[node] == {}:
            self._node_metadata[node] = metadata

    def add_nodes(self, node_list, metadata=None):
        """Add multiple nodes to the hypergraph."""
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidParameterError("node metadata must be a dict.")
        for node in node_list:
            try:
                self.add_node(node, metadata[node] if metadata is not None else None)
            except KeyError:
                raise InvalidParameterError(
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
            raise InvalidParameterError("Order and size cannot be both specified.")
        edges = [self._reverse_edge_list[edge_id] for edge_id in primary_adj[node]]
        return self._filter_edges_by_order(edges, order=order, size=size)

    def get_neighbors(self, node, order=None, size=None):
        """Return the set of neighbors of a node via incident edges."""
        if order is not None and size is not None:
            raise InvalidParameterError("Order and size cannot be both specified.")
        edges = self.get_incident_edges(node, order=order, size=size)
        neighbors = set()
        for edge_key in edges:
            neighbors.update(self._edge_nodes(edge_key))
        neighbors.discard(node)
        return neighbors

    def isolates(self, node_order=None):
        """Return isolated nodes.

        Parameters
        ----------
        node_order : list, optional
            If provided, return indices into this order instead of node labels.
        """

        def has_incidence(n):
            for adj in self._adjacency_maps().values():
                if n in adj and adj[n]:
                    return True
            return False

        isolated_nodes = [n for n in self.get_nodes() if not has_incidence(n)]
        if node_order is None:
            return isolated_nodes
        isolated_set = set(isolated_nodes)
        return [i for i, n in enumerate(node_order) if n in isolated_set]

    def non_isolates(self, node_order=None):
        """Return non-isolated nodes.

        Parameters
        ----------
        node_order : list, optional
            If provided, return indices into this order instead of node labels.
        """

        def has_incidence(n):
            for adj in self._adjacency_maps().values():
                if n in adj and adj[n]:
                    return True
            return False

        non_isolated_nodes = [n for n in self.get_nodes() if has_incidence(n)]
        if node_order is None:
            return non_isolated_nodes
        non_isolated_set = set(non_isolated_nodes)
        return [i for i, n in enumerate(node_order) if n in non_isolated_set]

    def incident_edges_by_node(self, index_by="edge_key", node_order=None):
        """Return incident edges for each node.

        Parameters
        ----------
        index_by : {"edge_key", "edge_id", "position"}
            Representation for incident edges.
        node_order : list, optional
            If provided, return a list aligned to this order.
        """
        if index_by not in {"edge_key", "edge_id", "position"}:
            raise ValueError(
                'index_by must be one of {"edge_key", "edge_id", "position"}.'
            )

        edge_list = list(self._edge_list.keys())
        pos_map = {edge: idx for idx, edge in enumerate(edge_list)}

        def map_edge_id(edge_id):
            if index_by == "edge_id":
                return edge_id
            edge_key = self._reverse_edge_list[edge_id]
            if index_by == "edge_key":
                return edge_key
            return pos_map[edge_key]

        result = {}
        for node in self.get_nodes():
            edge_ids = []
            for adj in self._adjacency_maps().values():
                if node in adj:
                    edge_ids.extend(adj[node])
            seen = set()
            deduped = []
            for edge_id in edge_ids:
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                deduped.append(map_edge_id(edge_id))
            if index_by == "position":
                deduped = sorted(deduped)
            result[node] = deduped

        if node_order is None:
            return result
        return [result[node] for node in node_order]

    def edges_by_size(self, index_by="edge_key"):
        """Return a dictionary mapping edge size to edges of that size.

        Parameters
        ----------
        index_by : {"edge_key", "edge_id", "position"}
            Representation for edges in the output.
        """
        if index_by not in {"edge_key", "edge_id", "position"}:
            raise ValueError(
                'index_by must be one of {"edge_key", "edge_id", "position"}.'
            )

        edges = list(self._edge_list.keys())
        edges_by_size = {}
        for idx, edge in enumerate(edges):
            size = self._edge_size(edge)
            if index_by == "edge_id":
                value = self._edge_list[edge]
            elif index_by == "position":
                value = idx
            else:
                value = edge
            edges_by_size.setdefault(size, []).append(value)
        return edges_by_size

    def edges_by_order(self, index_by="edge_key"):
        """Return a dictionary mapping edge order to edges of that order.

        Parameters
        ----------
        index_by : {"edge_key", "edge_id", "position"}
            Representation for edges in the output.
        """
        edges_by_size = self.edges_by_size(index_by=index_by)
        return {size - 1: edges for size, edges in edges_by_size.items()}

    def incidence_dict(
        self,
        axis="node",
        *,
        index_by="edge_key",
        node_order=None,
    ):
        """Return a dictionary representation of incidences.

        Parameters
        ----------
        axis : {"node", "edge"}
            If "node", map nodes to incident edges.
            If "edge", map edges to their nodes.
        index_by : {"edge_key", "edge_id", "position"}
            Representation for edges when axis="node".
        node_order : list, optional
            If provided, return node indices based on this ordering.
        """
        if axis not in {"node", "edge"}:
            raise ValueError('axis must be "node" or "edge".')

        if axis == "node":
            return self.incident_edges_by_node(index_by=index_by, node_order=node_order)

        if node_order is None:
            return {edge: list(self._edge_nodes(edge)) for edge in self._edge_list}
        index_map = {node: idx for idx, node in enumerate(node_order)}
        return {
            edge: [index_map[node] for node in self._edge_nodes(edge)]
            for edge in self._edge_list
        }

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
        if metadata is not None:
            self._validate_metadata_dict(metadata, "edge")
        if edge_key not in self._edge_list:
            edge_id = self._next_edge_id
            self._next_edge_id += 1
            self._edge_list[edge_key] = edge_id
            self._reverse_edge_list[edge_id] = edge_key
            self._weights[edge_id] = 1 if not self._weighted else weight
            self._edge_metadata[edge_id] = metadata or {}
            return edge_id

        duplicate_policy = self._duplicate_policy
        metadata_policy = self._metadata_policy

        if not self._weighted and duplicate_policy in {
            "accumulate_weight",
            "replace_weight",
        }:
            raise InvalidParameterError(
                "duplicate_policy must be 'ignore' or 'error' for unweighted hypergraphs."
            )

        edge_id = self._edge_list[edge_key]
        if duplicate_policy == "error":
            raise InvalidParameterError(f"Duplicate edge {edge_key} not allowed.")
        if duplicate_policy == "ignore":
            pass
        elif duplicate_policy == "accumulate_weight":
            if self._weighted:
                self._weights[edge_id] += weight
        elif duplicate_policy == "replace_weight":
            if self._weighted:
                self._weights[edge_id] = weight
        else:
            raise InvalidParameterError(
                "duplicate_policy must be one of: 'error', 'ignore', 'accumulate_weight', 'replace_weight'."
            )

        if metadata is not None:
            if metadata_policy == "replace":
                self._edge_metadata[edge_id] = metadata
            elif metadata_policy == "merge":
                self._edge_metadata[edge_id] = merge_metadata(
                    self._edge_metadata.get(edge_id), metadata
                )
            elif metadata_policy == "ignore":
                pass
            else:
                raise InvalidParameterError(
                    "metadata_policy must be one of: 'replace', 'merge', 'ignore'."
                )
        return edge_id

    def _add_edge(self, edge_key, weight=None, metadata=None):
        weight = self._validate_weight(weight)
        existed = edge_key in self._edge_list
        edge_id = self._add_edge_key(edge_key, weight=weight, metadata=metadata)
        if existed:
            return
        for node in self._edge_nodes(edge_key):
            self.add_node(node)
            self._add_incidence(node, edge_id, edge_key)

    def _remove_edge_key(self, edge_key):
        if edge_key not in self._edge_list:
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
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
            raise InvalidParameterError("Order and size cannot be both specified.")
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
            raise InvalidParameterError("Order and size cannot be both specified.")
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
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
        edge_id = self._edge_list[edge_key]
        self._weights[edge_id] = weight

    def get_weight(self, edge_key):
        if edge_key not in self._edge_list:
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
        edge_id = self._edge_list[edge_key]
        return self._weights[edge_id]

    def get_weights(self, order=None, size=None, up_to=False, asdict=False):
        """Return edge weights, optionally filtered by order or size."""
        w = None
        if order is not None and size is not None:
            raise InvalidParameterError("Order and size cannot be both specified.")
        if order is None and size is None:
            w = {edge: self._weights[self._edge_list[edge]] for edge in self._edge_list}
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
            raise InvalidParameterError("Order and size cannot be both specified.")
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
        self._validate_metadata_dict(metadata, "hypergraph")
        self._hypergraph_metadata = metadata

    def get_hypergraph_metadata(self):
        return self._hypergraph_metadata

    def set_node_metadata(self, node, metadata):
        if node not in self._node_metadata:
            self._raise_missing_node(node)
        self._validate_metadata_dict(metadata, "node")
        self._node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._node_metadata:
            self._raise_missing_node(node)
        return self._node_metadata[node]

    def get_all_nodes_metadata(self):
        return self._node_metadata

    def set_edge_metadata(self, edge_key, metadata):
        if edge_key not in self._edge_list:
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
        self._validate_metadata_dict(metadata, "edge")
        self._edge_metadata[self._edge_list[edge_key]] = metadata

    def get_edge_metadata(self, edge_key):
        if edge_key not in self._edge_list:
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
        return self._edge_metadata[self._edge_list[edge_key]]

    def get_all_edges_metadata(self):
        return self._edge_metadata

    def set_incidence_metadata(self, edge_key, node, metadata):
        if edge_key not in self._edge_list:
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
        self._validate_metadata_dict(metadata, "incidence")
        self._incidences_metadata[(edge_key, node)] = metadata

    def get_incidence_metadata(self, edge_key, node):
        if edge_key not in self._edge_list:
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
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
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
        self._edge_metadata[self._edge_list[edge_key]][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self._node_metadata:
            self._raise_missing_node(node)
        del self._node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge_key, field):
        if edge_key not in self._edge_list:
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
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

    def summary(
        self, *, include_size_distribution: bool = True, max_size_bins: int = 20
    ):
        """
        Lightweight summary for quick inspection.

        Returns a small dict suitable for printing/logging.
        """
        out = {
            "type": self._type_name(),
            "weighted": self._weighted,
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "uniform": self.is_uniform() if self.num_edges() else True,
        }

        if include_size_distribution and self.num_edges():
            dist = self.distribution_sizes()
            if len(dist) <= max_size_bins:
                out["size_distribution"] = dict(sorted(dist.items()))
            else:
                sizes = self.get_sizes()
                out["min_size"] = min(sizes) if sizes else None
                out["max_size"] = max(sizes) if sizes else None
                out["size_bins"] = len(dist)
        return out

    def __repr__(self):
        return "{}(nodes={}, edges={}, weighted={})".format(
            self._type_name(), self.num_nodes(), self.num_edges(), self._weighted
        )

    def __len__(self):
        return len(self._edge_list)

    def __iter__(self):
        """
        Iterate over internal edge storage.

        Notes
        -----
        This yields `(edge_key, edge_id)` pairs for backward compatibility with
        older code that relied on iterating the internal edge dictionary.

        For user-facing iteration, prefer:
        - `for node in H.nodes` / `H.iter_nodes()`
        - `for edge in H.edges` / `H.iter_edges()`
        """
        return iter(self._edge_list.items())

    def iter_nodes(self):
        """Iterate over node labels (user-facing)."""
        return iter(self.get_nodes())

    def iter_edges(self):
        """Iterate over edge keys (user-facing)."""
        return iter(self._edge_list.keys())

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
