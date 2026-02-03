from typing import List, Tuple

from hypergraphx.core.base import BaseHypergraph
from hypergraphx.exceptions import InvalidParameterError, MissingNodeError


class DirectedHypergraph(BaseHypergraph):
    """
    A Directed Hypergraph is a generalization of a graph in which hyperedges have a direction.
    Each hyperedge connects a set of source nodes to a set of target nodes.
    """

    def __init__(
        self,
        edge_list=None,
        weighted=True,
        weights=None,
        hypergraph_metadata=None,
        node_metadata=None,
        edge_metadata=None,
        duplicate_policy=None,
        metadata_policy=None,
    ):
        """
        Initialize a Directed Hypergraph.

        Parameters
        ----------
        edge_list : list of tuples of tuples, optional
            A list of directed hyperedges represented as (source_nodes, target_nodes),
            where source_nodes and target_nodes are tuples of nodes.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is True.
        weights : list of floats, optional
            A list of weights corresponding to the edges in `edge_list`. Required if `weighted` is True.
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
            If `edge_list` contains improperly formatted edges.
        """
        # Initialize hypergraph metadata
        self._adj_source = {}
        self._adj_target = {}
        metadata = hypergraph_metadata or {}
        metadata.update({"weighted": weighted, "type": "DirectedHypergraph"})
        self._init_base(
            weighted=weighted,
            hypergraph_metadata=metadata,
            node_metadata=node_metadata,
            duplicate_policy=duplicate_policy,
            metadata_policy=metadata_policy,
        )

        if edge_list is not None:
            if weighted and weights is not None and len(edge_list) != len(weights):
                raise ValueError("Edge list and weights must have the same length.")
            self.add_edges(edge_list, weights=weights, metadata=edge_metadata)

    def _adjacency_maps(self):
        return {"source": self._adj_source, "target": self._adj_target}

    def _init_node_adjacency(self, node):
        self._adj_source[node] = []
        self._adj_target[node] = []

    def _normalize_edge(self, edge, **kwargs):
        source, target = edge
        try:
            source = tuple(sorted(tuple(source)))
        except TypeError:
            source = tuple(sorted((source,)))
        try:
            target = tuple(sorted(tuple(target)))
        except TypeError:
            target = tuple(sorted((target,)))
        return (source, target)

    def _edge_nodes(self, edge_key):
        return tuple(edge_key[0]) + tuple(edge_key[1])

    def _edge_size(self, edge_key):
        return len(edge_key[0]) + len(edge_key[1])

    def _edge_key_without_node(self, edge_key, node):
        source, target = edge_key
        return (
            tuple(n for n in source if n != node),
            tuple(n for n in target if n != node),
        )

    def _add_edge(
        self,
        edge_key,
        weight=None,
        metadata=None,
    ):
        weight = self._validate_weight(weight)
        existed = edge_key in self._edge_list
        edge_id = self._add_edge_key(edge_key, weight=weight, metadata=metadata)
        if existed:
            return
        source, target = edge_key
        for node in source:
            self.add_node(node)
            self._adj_source[node].append(edge_id)
        for node in target:
            self.add_node(node)
            self._adj_target[node].append(edge_id)

    def _new_like(self):
        return DirectedHypergraph(weighted=self._weighted)

    def _hash_edge_nodes(self, edge_key):
        return (tuple(sorted(edge_key[0])), tuple(sorted(edge_key[1])))

    # Nodes
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
        if metadata is None:
            metadata = {}
        if node not in self._adj_source:
            self._adj_source[node] = []
            self._adj_target[node] = []
            self._node_metadata[node] = {}
        if self._node_metadata[node] == {}:
            self._node_metadata[node] = metadata

    def add_nodes(self, node_list: list, metadata=None):
        """
        Add a list of nodes to the hypergraph.

        Parameters
        ----------
        node_list : list
            The list of nodes to add.
        metadata : dict, optional
            Optional mapping of nodes to metadata.

        Returns
        -------
        None
        """
        for node in node_list:
            try:
                self.add_node(node, metadata[node] if metadata is not None else None)
            except KeyError:
                raise ValueError(
                    "The metadata dictionary must contain an entry for each node in the node list."
                )

    def remove_node(self, node, keep_edges=False):
        """Remove a node from the hypergraph, with an option to keep or remove edges incident to it."""
        if node not in self._adj_source or node not in self._adj_target:
            self._raise_missing_node(node)

        # Handle incident edges
        edge_ids = set(self._adj_source[node]) | set(self._adj_target[node])
        if keep_edges:
            for edge_id in list(edge_ids):
                edge_key = self._reverse_edge_list[edge_id]
                updated_key = self._edge_key_without_node(edge_key, node)
                weight = self._weights.get(edge_id, 1)
                metadata = self._edge_metadata.get(edge_id, {})
                self._remove_edge_key(edge_key)
                if self._allow_empty_edge() or self._edge_size(updated_key) > 0:
                    self._add_edge(updated_key, weight=weight, metadata=metadata)
        else:
            for edge_id in list(edge_ids):
                edge_key = self._reverse_edge_list[edge_id]
                self._remove_edge_key(edge_key)

        del self._adj_source[node]
        del self._adj_target[node]
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
            If True, the edges incident to the nodes are kept, but the nodes are removed from the edges. If False, the edges incident to the nodes are removed. Default is False.


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
        """Returns the list of nodes in the hypergraph."""
        if not metadata:
            return list(self._adj_source.keys())
        return {node: self._node_metadata[node] for node in self._adj_source.keys()}

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
        return node in self._adj_source or node in self._adj_target

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
        if node not in self._adj_source.keys():
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        if node not in self._adj_target.keys():
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        if order is not None and size is not None:
            raise InvalidParameterError("Order and size cannot be both specified.")
        if order is None and size is None:
            neigh = set()
            edges = self.get_incident_edges(node)
            for source, target in edges:
                neigh.update(source)
                neigh.update(target)
            if node in neigh:
                neigh.remove(node)
            return neigh
        else:
            if order is None:
                order = size - 1
            neigh = set()
            edges = self.get_incident_edges(node, order=order)
            for source, target in edges:
                neigh.update(source)
                neigh.update(target)
            if node in neigh:
                neigh.remove(node)
            return neigh

    def get_incident_edges(self, node, order: int = None, size: int = None):
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
            raise InvalidParameterError("Order and size cannot be both specified.")
        if order is None and size is None:
            return self.get_source_edges(node) + self.get_target_edges(node)
        elif size is not None:
            return self.get_source_edges(node, size=size) + self.get_target_edges(
                node, size=size
            )
        elif order is not None:
            return self.get_source_edges(node, order=order) + self.get_target_edges(
                node, order=order
            )

    def get_sources(self):
        """Returns the list of sources of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of sources of the hyperedges in the hypergraph.

        """
        return [edge[0] for edge in self._edge_list.keys()]

    def get_targets(self):
        """Returns the list of targets of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of targets of the hyperedges in the hypergraph.

        """
        return [edge[1] for edge in self._edge_list.keys()]

    # Edges
    def add_edge(
        self,
        edge: Tuple[Tuple, Tuple],
        weight=None,
        metadata=None,
    ):
        """Add a directed hyperedge to the hypergraph. If the hyperedge already exists, its weight is updated.

        Parameters
        ----------
        edge : tuple of tuples
            The directed hyperedge to add, represented as (source_nodes, target_nodes).
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
        No multi-edges: duplicates never create a new edge. Control behavior via the
        hypergraph-level policies:
        - `set_duplicate_policy(...)`
        - `set_metadata_policy(...)`

        Incidence metadata is not modified by duplicate adds; use incidence-metadata APIs explicitly.
        """
        edge_key = self._normalize_edge(edge)
        self._add_edge(edge_key, weight=weight, metadata=metadata)

    def add_edges(
        self,
        edge_list: List[Tuple[Tuple, Tuple]],
        weights=None,
        metadata=None,
    ):
        """Add a list of directed hyperedges to the hypergraph. If a hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge_list : list of tuples of tuples
            The list of directed hyperedges to add (each as (source_nodes, target_nodes)).
        weights : list, optional
            The list of weights of the hyperedges.
        metadata : list, optional
            The list of metadata of the hyperedges.

        Returns
        -------
        None
        Notes
        -----
        No multi-edges: duplicates never create a new edge. See `add_edge()` for policies.
        """
        if weights is not None:
            if len(edge_list) != len(weights):
                raise ValueError("The number of edges and weights must be the same.")
            if not self._weighted:
                for weight in weights:
                    if weight not in (None, 1):
                        raise ValueError(
                            "If the hypergraph is not weighted, weight can be 1 or None."
                        )

        for i, edge in enumerate(edge_list):
            self.add_edge(
                edge,
                weight=weights[i] if weights else None,
                metadata=metadata[i] if metadata else None,
            )

    def get_source_edges(self, node, order=None, size=None):
        """
        Get the source edges in which a node is in the source set.

        Parameters
        ----------
        node
        order
        size

        Returns
        -------
        list
            The list of incident in-edges.
        """
        if node not in self._adj_source:
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        if order is not None and size is not None:
            raise InvalidParameterError("Order and size cannot be both specified.")
        if order is None and size is None:
            return [self._reverse_edge_list[e_idx] for e_idx in self._adj_source[node]]
        if size is not None:
            return [
                self._reverse_edge_list[e_idx]
                for e_idx in self._adj_source[node]
                if self._edge_size(self._reverse_edge_list[e_idx]) == size
            ]
        return [
            self._reverse_edge_list[e_idx]
            for e_idx in self._adj_source[node]
            if self._edge_size(self._reverse_edge_list[e_idx]) - 1 == order
        ]

    def get_target_edges(self, node, order=None, size=None):
        """
        Get the hyperedges in which a node is in the target set.

        Parameters
        ----------
        node
        order
        size

        Returns
        -------
        list
            The list of incident out-edges.
        """
        if node not in self._adj_target:
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        if order is not None and size is not None:
            raise InvalidParameterError("Order and size cannot be both specified.")
        if order is None and size is None:
            return [self._reverse_edge_list[e_idx] for e_idx in self._adj_target[node]]
        if size is not None:
            return [
                self._reverse_edge_list[e_idx]
                for e_idx in self._adj_target[node]
                if self._edge_size(self._reverse_edge_list[e_idx]) == size
            ]
        return [
            self._reverse_edge_list[e_idx]
            for e_idx in self._adj_target[node]
            if self._edge_size(self._reverse_edge_list[e_idx]) - 1 == order
        ]

    def get_edges(
        self,
        order=None,
        size=None,
        up_to=False,
        subhypergraph=False,
        keep_isolated_nodes=False,
        metadata=False,
    ):
        return self._get_edges_common(
            order=order,
            size=size,
            up_to=up_to,
            subhypergraph=subhypergraph,
            keep_isolated_nodes=keep_isolated_nodes,
            metadata=metadata,
        )

    def remove_edge(self, edge: Tuple[Tuple, Tuple]):
        """Remove a directed edge from the hypergraph.

        Parameters
        ----------
        edge : tuple of tuples
            The edge to remove (source_nodes, target_nodes).

        Returns
        -------
        None
        """
        edge_key = self._normalize_edge(edge)
        self._remove_edge_key(edge_key)

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

    def set_edge_list(self, edge_list):
        self._guard_unsafe_setter("DirectedHypergraph.set_edge_list")
        self._edge_list = edge_list
        self._maybe_validate_invariants()

    def get_edge_list(self):
        return self._edge_list

    def check_edge(self, edge: Tuple[Tuple, Tuple]):
        """Checks if the specified edge is in the hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to check.

        Returns
        -------
        bool
            True if the edge is in the hypergraph, False otherwise.

        """
        edge_key = self._normalize_edge(edge)
        return self._edge_exists(edge_key)

    # Weight
    def get_weight(self, edge: Tuple[Tuple, Tuple]):
        """Returns the weight of the specified directed edge."""
        edge_key = self._normalize_edge(edge)
        return super().get_weight(edge_key)

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
        return super().get_weights(order=order, size=size, up_to=up_to, asdict=asdict)

    def set_weight(self, edge: Tuple[Tuple, Tuple], weight: float):
        """Sets the weight of the specified directed edge."""
        edge_key = self._normalize_edge(edge)
        super().set_weight(edge_key, weight)

    # Info
    def is_uniform(self):
        """
        Check if the hypergraph is uniform, i.e. all hyperedges have the same size.

        Returns
        -------
        bool
            True if the hypergraph is uniform, False otherwise.
        """
        uniform = True
        sz = None
        for edge in self._edge_list:
            edge_nodes = set(edge[0]).union(set(edge[1]))
            if sz is None:
                sz = len(edge_nodes)
            elif len(edge_nodes) != sz:
                uniform = False
                break
        return uniform

    # Adj
    def get_adj_dict(self, source_target):
        if source_target == "source":
            return self._adj_source
        elif source_target == "target":
            return self._adj_target
        else:
            raise ValueError(
                "Invalid value for source_target. Must be 'source' or 'target'."
            )

    def set_adj_dict(self, adj_dict, source_target):
        self._guard_unsafe_setter("DirectedHypergraph.set_adj_dict")
        if source_target == "source":
            self._adj_source = adj_dict
        elif source_target == "target":
            self._adj_target = adj_dict
        else:
            raise ValueError(
                "Invalid value for source_target. Must be 'source' or 'target'."
            )
        self._maybe_validate_invariants()

    # Degree
    def degree(self, node, order=None, size=None):
        from hypergraphx.measures.degree import degree

        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_sequence

        return degree_sequence(self, order=order, size=size)

    def degree_distribution(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_distribution

        return degree_distribution(self, order=order, size=size)

    def in_degree(self, node, order=None, size=None):
        """Return the in-degree of a node, counting incident edges where the node is a target.

        Parameters
        ----------
        node : object
            The node of interest.
        order : int, optional
            The order of the hyperedges to consider.
        size : int, optional
            The size of the hyperedges to consider.
        """
        return len(self.get_target_edges(node, order=order, size=size))

    def out_degree(self, node, order=None, size=None):
        """Return the out-degree of a node, counting incident edges where the node is a source.

        Parameters
        ----------
        node : object
            The node of interest.
        order : int, optional
            The order of the hyperedges to consider.
        size : int, optional
            The size of the hyperedges to consider.
        """
        return len(self.get_source_edges(node, order=order, size=size))

    def in_degree_sequence(self, order=None, size=None):
        """Return the in-degree for every node as a dict."""
        return {
            node: self.in_degree(node, order=order, size=size)
            for node in self.get_nodes()
        }

    def out_degree_sequence(self, order=None, size=None):
        """Return the out-degree for every node as a dict."""
        return {
            node: self.out_degree(node, order=order, size=size)
            for node in self.get_nodes()
        }

    def in_degree_distribution(self, order=None, size=None):
        """Return a histogram of in-degrees as a dict {degree: count}."""
        dist = {}
        for node, deg in self.in_degree_sequence(order=order, size=size).items():
            dist[deg] = dist.get(deg, 0) + 1
        return dist

    def out_degree_distribution(self, order=None, size=None):
        """Return a histogram of out-degrees as a dict {degree: count}."""
        dist = {}
        for node, deg in self.out_degree_sequence(order=order, size=size).items():
            dist[deg] = dist.get(deg, 0) + 1
        return dist

    # Utility
    def isolated_nodes(self, size=None, order=None):
        from hypergraphx.utils.components import isolated_nodes

        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.components import is_isolated

        return is_isolated(self, node, size=size, order=order)

    def to_line_graph(self, distance="intersection", s: int = 1, weighted=False):
        from hypergraphx.representations.projections import directed_line_graph

        return directed_line_graph(self, distance, s, weighted)

    def to_hypergraph(
        self,
        keep_node_metadata: bool = True,
        keep_edge_metadata: bool = True,
        keep_hypergraph_metadata: bool = True,
    ):
        """Convert to an undirected Hypergraph by merging sources and targets.

        Duplicate hyperedges are merged by summing weights and merging metadata.
        """
        from hypergraphx.core.undirected import Hypergraph
        from hypergraphx.utils.metadata import merge_metadata

        hg = Hypergraph(weighted=True)
        if keep_hypergraph_metadata:
            meta = merge_metadata(
                self.get_hypergraph_metadata(), {"converted_from": "DirectedHypergraph"}
            )
            hg.set_hypergraph_metadata(meta)

        if keep_node_metadata:
            for node, metadata in self.get_all_nodes_metadata().items():
                hg.add_node(node, metadata=metadata)

        edge_weights = {}
        edge_metadata = {}
        for edge in self.get_edges():
            source, target = edge
            merged_edge = tuple(sorted(set(source).union(target)))
            edge_weights[merged_edge] = edge_weights.get(
                merged_edge, 0
            ) + self.get_weight(edge)
            if keep_edge_metadata:
                edge_metadata[merged_edge] = merge_metadata(
                    edge_metadata.get(merged_edge), self.get_edge_metadata(edge)
                )

        for edge, weight in edge_weights.items():
            hg.add_edge(edge, weight=weight, metadata=edge_metadata.get(edge))

        return hg

    # Metadata
    def get_all_nodes_metadata(self):
        return self._node_metadata

    def set_edge_metadata(self, edge, metadata):
        edge_key = self._normalize_edge(edge)
        super().set_edge_metadata(edge_key, metadata)

    def get_edge_metadata(self, edge):
        edge_key = self._normalize_edge(edge)
        return super().get_edge_metadata(edge_key)

    def get_all_edges_metadata(self):
        return self._edge_metadata

    def set_incidence_metadata(self, edge, node, metadata):
        edge_key = self._normalize_edge(edge)
        super().set_incidence_metadata(edge_key, node, metadata)

    def get_incidence_metadata(self, edge, node):
        edge_key = self._normalize_edge(edge)
        return super().get_incidence_metadata(edge_key, node)

    def get_all_incidences_metadata(self):
        return {k: v for k, v in self._incidences_metadata.items()}

    def set_attr_to_hypergraph_metadata(self, field, value):
        self._hypergraph_metadata[field] = value

    def set_attr_to_node_metadata(self, node, field, value):
        super().set_attr_to_node_metadata(node, field, value)

    def set_attr_to_edge_metadata(self, edge, field, value):
        edge_key = self._normalize_edge(edge)
        super().set_attr_to_edge_metadata(edge_key, field, value)

    def remove_attr_from_node_metadata(self, node, field):
        super().remove_attr_from_node_metadata(node, field)

    def remove_attr_from_edge_metadata(self, edge, field):
        edge_key = self._normalize_edge(edge)
        super().remove_attr_from_edge_metadata(edge_key, field)

    # Basic Functions
    def clear(self):
        super().clear()

    # Data Structure Extra
    def populate_from_dict(self, data):
        """
        Populate the attributes of the directed hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        super().populate_from_dict(data)
