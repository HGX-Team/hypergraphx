from hypergraphx.core.base import BaseHypergraph
from hypergraphx.exceptions import InvalidParameterError, MissingEdgeError


class Hypergraph(BaseHypergraph):
    """
    A Hypergraph is a generalization of a graph where an edge (hyperedge) can connect
    any number of nodes. It is represented as a set of nodes and a set of hyperedges,
    where each hyperedge is a subset of nodes.
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
        Initialize a Hypergraph.

        Parameters
        ----------
        edge_list : list of tuples, optional
            A list of hyperedges, where each hyperedge is represented as a tuple of nodes.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is True.
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
        self._adj = {}
        self._empty_edges = {}
        metadata = hypergraph_metadata or {}
        metadata.update({"weighted": weighted, "type": "Hypergraph"})
        self._init_base(
            weighted=weighted,
            hypergraph_metadata=metadata,
            node_metadata=node_metadata,
            duplicate_policy=duplicate_policy,
            metadata_policy=metadata_policy,
        )

        if edge_list:
            if weighted and weights is not None and len(edge_list) != len(weights):
                raise ValueError("Edge list and weights must have the same length.")
            self.add_edges(edge_list, weights=weights, metadata=edge_metadata)

    def _normalize_edge(self, edge, **kwargs):
        return tuple(sorted(edge))

    def _edge_nodes(self, edge_key):
        return edge_key

    def _allow_empty_edge(self):
        return True

    def _new_like(self):
        return Hypergraph(weighted=self._weighted)

    # Nodes
    def remove_node(self, node, keep_edges=False):
        """Remove a node from the hypergraph.

        Parameters
        ----------
        node
            The node to remove.
        keep_edges : bool, optional
            If True, the edges incident to the node are kept, but the node is removed from the edges. If False, the edges incident to the node are removed. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the node is not in the hypergraph.
        """
        if node not in self._adj:
            self._raise_missing_node(node)
        super().remove_node(node, keep_edges=keep_edges)

    # Edges
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
        Notes
        -----
        No multi-edges: duplicates never create a new edge. See `add_edge()` for policies.

        """
        if edge_list is not None:
            edge_list = list(edge_list)
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
                    weight=(
                        weights[i] if self._weighted and weights is not None else None
                    ),
                    metadata=metadata[i] if metadata is not None else None,
                )

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
        edge_key = self._normalize_edge(edge)
        if not self._edge_exists(edge_key):
            raise MissingEdgeError(f"Edge {edge_key} not in hypergraph.")
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
        self._guard_unsafe_setter("Hypergraph.set_edge_list")
        self._edge_list = edge_list
        self._maybe_validate_invariants()

    def get_edge_list(self):
        return self._edge_list

    def add_empty_edge(self, name, metadata):
        if name not in self._empty_edges:
            self._empty_edges[name] = metadata
        else:
            raise ValueError("Edge {} already in hypergraph.".format(name))

    def check_edge(self, edge):
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
        return self._edge_exists(self._normalize_edge(edge))

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

    # Weight
    def set_weight(self, edge, weight):
        edge_key = self._normalize_edge(edge)
        super().set_weight(edge_key, weight)

    def get_weight(self, edge):
        edge_key = self._normalize_edge(edge)
        return super().get_weight(edge_key)

    # Info
    # Adj And Subhypergraph
    def get_adj_dict(self):
        return self._adj

    def set_adj_dict(self, adj):
        self._guard_unsafe_setter("Hypergraph.set_adj_dict")
        self._adj = adj
        self._maybe_validate_invariants()

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
        nodes_set = set(nodes)
        for node in nodes:
            h.set_node_metadata(node, self.get_node_metadata(node))

        candidate_edge_ids = set()
        for node in nodes:
            candidate_edge_ids.update(self._adj.get(node, []))

        for edge_id in candidate_edge_ids:
            edge = self._reverse_edge_list[edge_id]
            if nodes_set.issuperset(edge):
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
        """Return a subhypergraph induced by the edges of the specified orders.

        Parameters
        ----------
        orders : list, optional
            List of orders of the edges to be included in the subhypergraph. If None, the sizes parameter should be specified.
        sizes : list, optional
            List of sizes of the edges to be included in the subhypergraph. If None, the orders parameter should be specified.
        keep_nodes : bool, optional
            If True, the nodes of the original hypergraph are kept in the subhypergraph. If False, only the edges are kept. Default is True.

        Returns
        -------
        Hypergraph
            Subhypergraph induced by the edges of the specified orders.

        Raises
        ------
        ValueError
            If both orders and sizes are None or if both orders and sizes are specified.
        """
        if orders is None and sizes is None:
            raise InvalidParameterError(
                "At least one between orders and sizes should be specified"
            )
        if orders is not None and sizes is not None:
            raise InvalidParameterError("Order and size cannot be both specified.")
        h = Hypergraph(weighted=self.is_weighted())
        if keep_nodes:
            h.add_nodes(node_list=list(self.get_nodes()))
            for node in self.get_nodes():
                h.set_node_metadata(node, self.get_node_metadata(node))

        if sizes is None:
            sizes = []
            for order in orders:
                sizes.append(order + 1)

        for size in sizes:
            edges = self.get_edges(size=size)
            for edge in edges:
                if h.is_weighted():
                    h.add_edge(
                        edge, self.get_weight(edge), self.get_edge_metadata(edge)
                    )
                else:
                    h.add_edge(edge, metadata=self.get_edge_metadata(edge))

        return h

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

    # Connected Components
    def is_connected(self, size=None, order=None):
        from hypergraphx.utils.components import is_connected

        return is_connected(self, size=size, order=order)

    def connected_components(self, size=None, order=None):
        from hypergraphx.utils.components import connected_components

        return connected_components(self, size=size, order=order)

    def node_connected_component(self, node, size=None, order=None):
        from hypergraphx.utils.components import node_connected_component

        return node_connected_component(self, node, size=size, order=order)

    def num_connected_components(self, size=None, order=None):
        from hypergraphx.utils.components import num_connected_components

        return num_connected_components(self, size=size, order=order)

    def largest_component(self, size=None, order=None):
        from hypergraphx.utils.components import largest_component

        return largest_component(self, size=size, order=order)

    def subhypergraph_largest_component(self, size=None, order=None):
        """
        Returns a subhypergraph induced by the nodes in the largest component of the hypergraph.

        Parameters
        ----------
        size: int, optional
            The size of the hyperedges to consider
        order: int, optional
            The order of the hyperedges to consider

        Returns
        -------
        Hypergraph
            Subhypergraph induced by the nodes in the largest component of the hypergraph.
        """
        nodes = self.largest_component(size=size, order=order)
        return self.subhypergraph(nodes)

    def largest_component_size(self, size=None, order=None):
        from hypergraphx.utils.components import largest_component_size

        return largest_component_size(self, size=size, order=order)

    # Matrix
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
        from hypergraphx.utils.components import isolated_nodes

        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.components import is_isolated

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

    def set_edge_metadata(self, edge, metadata):
        edge_key = self._normalize_edge(edge)
        super().set_edge_metadata(edge_key, metadata)

    def get_edge_metadata(self, edge):
        edge_key = self._normalize_edge(edge)
        return super().get_edge_metadata(edge_key)

    def set_incidence_metadata(self, edge, node, metadata):
        edge_key = self._normalize_edge(edge)
        super().set_incidence_metadata(edge_key, node, metadata)

    def get_incidence_metadata(self, edge, node):
        edge_key = self._normalize_edge(edge)
        return super().get_incidence_metadata(edge_key, node)

    # Basic Functions
    def clear(self):
        super().clear()
        self._empty_edges.clear()

    # Data Structure Extra
    def populate_from_dict(self, data):
        """
        Populate the attributes of the hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        super().populate_from_dict(data)
