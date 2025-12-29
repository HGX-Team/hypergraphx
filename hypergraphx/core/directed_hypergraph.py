import warnings
from typing import List, Tuple

from hypergraphx.core.base_hypergraph import BaseHypergraph


class DirectedHypergraph(BaseHypergraph):
    """
    A Directed Hypergraph is a generalization of a graph in which hyperedges have a direction.
    Each hyperedge connects a set of source nodes to a set of target nodes.
    """

    def __init__(
        self,
        edge_list=None,
        weighted=False,
        weights=None,
        hypergraph_metadata=None,
        node_metadata=None,
        edge_metadata=None,
    ):
        """
        Initialize a Directed Hypergraph.

        Parameters
        ----------
        edge_list : list of tuples of tuples, optional
            A list of directed hyperedges represented as (source_nodes, target_nodes),
            where source_nodes and target_nodes are tuples of nodes.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is False.
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

    def _add_edge(self, edge_key, weight=None, metadata=None):
        weight = self._validate_weight(weight)
        edge_id = self._add_edge_key(edge_key, weight=weight, metadata=metadata)
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

    def add_nodes(self, node_list: list):
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
        for node in node_list:
            self.add_node(node)

    def remove_node(self, node, keep_edges=False):
        """Remove a node from the hypergraph, with an option to keep or remove edges incident to it."""
        if node not in self._adj_source or node not in self._adj_target:
            raise KeyError(f"Node {node} not in hypergraph.")

        # Handle incident edges
        if not keep_edges:
            target_edges = self.get_target_edges(node)
            source_edges = self.get_source_edges(node)
            for edge in source_edges:
                self.remove_edge(edge)
            for edge in target_edges:
                self.remove_edge(edge)

        del self._adj_source[node]
        del self._adj_target[node]

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
            raise ValueError("Node {} not in hypergraph.".format(node))
        if node not in self._adj_target.keys():
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
            raise ValueError("Order and size cannot be both specified.")
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
    def add_edge(self, edge: Tuple[Tuple, Tuple], weight=None, metadata=None):
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
        """
        edge_key = self._normalize_edge(edge)
        self._add_edge(edge_key, weight=weight, metadata=metadata)

    def add_edges(
        self, edge_list: List[Tuple[Tuple, Tuple]], weights=None, metadata=None
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
        """
        if weights is not None and not self._weighted:
            warnings.warn(
                "Weights are provided but the hypergraph is not weighted. The hypergraph will be weighted.",
                UserWarning,
            )
            self._weighted = True

        if self._weighted and weights is not None:
            if len(edge_list) != len(weights):
                raise ValueError("The number of edges and weights must be the same.")

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
            raise ValueError(f"Node {node} not in hypergraph.")
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
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
            raise ValueError(f"Node {node} not in hypergraph.")
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
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
        self._edge_list = edge_list

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
        return super().get_weights(
            order=order, size=size, up_to=up_to, asdict=asdict
        )

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
        if source_target == "source":
            self._adj_source = adj_dict
        elif source_target == "target":
            self._adj_target = adj_dict
        else:
            raise ValueError(
                "Invalid value for source_target. Must be 'source' or 'target'."
            )

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
    
    # Utility
    def isolated_nodes(self, size=None, order=None):
        from hypergraphx.utils.cc import isolated_nodes

        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.cc import is_isolated

        return is_isolated(self, node, size=size, order=order)

    def to_line_graph(self, distance="intersection", s: int = 1, weighted=False):
        from hypergraphx.representations.projections import directed_line_graph

        return directed_line_graph(self, distance, s, weighted)

    # Metadata
    def get_all_nodes_metadata(self):
        return list(self._node_metadata.values())

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
