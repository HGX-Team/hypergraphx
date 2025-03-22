import copy
import warnings

from sklearn.preprocessing import LabelEncoder


class Hypergraph:
    """
    A Hypergraph is a generalization of a graph where an edge (hyperedge) can connect
    any number of nodes. It is represented as a set of nodes and a set of hyperedges,
    where each hyperedge is a subset of nodes.
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
        # Initialize hypergraph metadata
        self._hypergraph_metadata = hypergraph_metadata or {}
        self._hypergraph_metadata.update({"weighted": weighted, "type": "Hypergraph"})

        # Initialize core attributes
        self._weighted = weighted
        self._adj = {}
        self._edge_list = {}
        self._weights = {}
        self._incidences_metadata = {}
        self._node_metadata = {}
        self._edge_metadata = {}
        self._empty_edges = {}
        self._reverse_edge_list = {}
        self._next_edge_id = 0

        # Add node metadata if provided
        if node_metadata:
            for node, metadata in node_metadata.items():
                self.add_node(node, metadata=metadata)

        # Add edges if provided
        if edge_list:
            if weighted and weights is not None and len(edge_list) != len(weights):
                raise ValueError("Edge list and weights must have the same length.")
            self.add_edges(edge_list, weights=weights, metadata=edge_metadata)
    #Nodes
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
        if node not in self._adj:
            self._adj[node] = []
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
        """
        Returns the list of nodes in the hypergraph. If metadata is True, it returns a list of tuples (node, metadata).

        Parameters
        ----------
        metadata : bool, optional

        Returns
        -------
        list
            List of nodes in the hypergraph. If metadata is True, it returns a list of tuples (node, metadata).
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

    #Edges
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

        edge = tuple(sorted(edge))
        order = len(edge) - 1
        if metadata is None:
            metadata = {}

        if edge not in self._edge_list:
            self._edge_list[edge] = self._next_edge_id
            self._reverse_edge_list[self._next_edge_id] = edge
            self._weights[self._next_edge_id] = 1 if not self._weighted else weight
            self._next_edge_id += 1
        elif edge in self._edge_list and self._weighted:
            self._weights[self._edge_list[edge]] += weight

        if metadata is not None:
            self._edge_metadata[self._edge_list[edge]] = metadata
        else:
            self._edge_metadata[self._edge_list[edge]] = {}

        for node in edge:
            self.add_node(node)
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
        edge = tuple(sorted(edge))
        if edge not in self._edge_list:
            raise KeyError("Edge {} not in hypergraph.".format(edge))

        for node in edge:
            try:
                self._adj[node].remove(self._edge_list[edge])
            except KeyError:
                pass

        del self._reverse_edge_list[self._edge_list[edge]]
        del self._edge_metadata[self._edge_list[edge]]
        del self._weights[self._edge_list[edge]]
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

    def set_edge_list(self, edge_list):
        self._edge_list = edge_list

    def get_edge_list(self):
        return self._edge_list

    def add_empty_edge(self, name, metadata):
        if name not in self._empty_edges:
            self._empty_edges[name] = metadata
        else:
            raise ("Edge {} already in hypergraph.".format(name))

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
        return tuple(sorted(edge)) in self._edge_list

    def get_edges(
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

        if subhypergraph and keep_isolated_nodes:
            h = Hypergraph(weighted=self._weighted)
            h.add_nodes(list(self.get_nodes()))
            if self._weighted:
                edge_weights = [self.get_weight(edge) for edge in edges]
                h.add_edges(edges, edge_weights)
            else:
                h.add_edges(edges)

            for node in h.get_nodes():
                h.set_node_metadata(node, self.get_node_metadata(node))
            for edge in edges:
                h.set_edge_metadata(edge, self.get_edge_metadata(edge))
            return h
        elif subhypergraph:
            h = Hypergraph(weighted=self._weighted)
            if self._weighted:
                edge_weights = [self.get_weight(edge) for edge in edges]
                h.add_edges(edges, edge_weights)
            else:
                h.add_edges(edges)

            for edge in edges:
                h.set_edge_metadata(edge, self.get_edge_metadata(edge))
            return h
        else:
            return (
                edges
                if not metadata
                else {edge: self.get_edge_metadata(edge) for edge in edges}
            )

    #Weight
    def set_weight(self, edge, weight):
        """Sets the weight of the specified edge.

        Parameters
        ----------
        edge : tuple
            The edge to set the weight of.

        weight : float
            The weight to set.

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

        edge = tuple(sorted(edge))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        edge_id = self._edge_list[edge]
        self._weights[edge_id] = weight

    def get_weight(self, edge):
        """Returns the weight of the specified edge.

        Parameters
        ----------
        edge : tuple
            The edge to get the weight of.

        Returns
        -------
        float
            Weight of the specified edge.
        """
        edge = tuple(sorted(edge))
        if edge in self._edge_list:
            edge_id = self._edge_list[edge]
            return self._weights[edge_id]
        else:
            raise ValueError("Edge {} not in hypergraph.".format(edge))

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
        w = None
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            w = {
                edge: self._weights[self._edge_list[edge]] for edge in self.get_edges()
            }

        if size is not None:
            order = size - 1

        if w is None:
            w = {
                edge: self._weights[self._edge_list[edge]]
                for edge in self.get_edges(order=order, up_to=up_to)
            }

        if asdict:
            return w
        else:
            return list(w.values())

    #Info
    def max_order(self):
        """
        Returns the maximum order of the hypergraph.

        Returns
        -------
        int
            Maximum order of the hypergraph.
        """
        return self.max_size() - 1

    def max_size(self):
        """
        Returns the maximum size of the hypergraph.

        Returns
        -------
        int
            Maximum size of the hypergraph.
        """
        return max(self.get_sizes())

    def num_nodes(self):
        """
        Returns the number of nodes in the hypergraph.

        Returns
        -------
        int
            Number of nodes in the hypergraph.
        """
        return len(list(self.get_nodes()))

    def num_edges(self, order=None, size=None, up_to=False):
        """Returns the number of edges in the hypergraph. If order is specified, it returns the number of edges of the specified order.
        If size is specified, it returns the number of edges of the specified size. If both order and size are specified, it raises a ValueError.
        If up_to is True, it returns the number of edges of order smaller or equal to the specified order.

        Parameters
        ----------
        order : int, optional
            Order of the edges to count.
        size : int, optional
            Size of the edges to count.
        up_to : bool, optional
            If True, it returns the number of edges of order smaller or equal to the specified order. Default is False.

        Returns
        -------
        int
            Number of edges in the hypergraph.
        """
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")

        if order is None and size is None:
            return len(self._edge_list)
        else:
            if size is not None:
                order = size - 1
            if not up_to:
                s = 0
                for edge in self._edge_list:
                    if len(edge) - 1 == order:
                        s += 1
                return s
            else:
                s = 0
                for edge in self._edge_list:
                    if len(edge) - 1 <= order:
                        s += 1
                return s

    def get_sizes(self):
        """Returns the list of sizes of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of sizes of the hyperedges in the hypergraph.

        """
        return [len(edge) for edge in self._edge_list.keys()]

    def distribution_sizes(self):
        """
        Returns the distribution of sizes of the hyperedges in the hypergraph.

        Returns
        -------
        collections.Counter
            Distribution of sizes of the hyperedges in the hypergraph.
        """
        from collections import Counter

        return dict(Counter(self.get_sizes()))

    def get_orders(self):
        """Returns the list of orders of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of orders of the hyperedges in the hypergraph.

        """
        return [len(edge) - 1 for edge in self._edge_list.keys()]

    def is_weighted(self):
        """
        Check if the hypergraph is weighted.

        Returns
        -------
        bool
            True if the hypergraph is weighted, False otherwise.
        """
        return self._weighted

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
            if sz is None:
                sz = len(edge)
            else:
                if len(edge) != sz:
                    uniform = False
                    break
        return uniform

    #Adj And Subhypergraph
    def get_adj_dict(self):
        return self._adj

    def set_adj_dict(self, adj):
        self._adj = adj

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
                        weight=self._edge_list[edge],
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

    #Degree
    def degree(self, node, order=None, size=None):
        from hypergraphx.measures.degree import degree

        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_sequence

        return degree_sequence(self, order=order, size=size)

    def degree_distribution(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_distribution

        return degree_distribution(self, order=order, size=size)

    #Connected Components
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
        from hypergraphx.utils.cc import largest_component_size

        return largest_component_size(self, size=size, order=order)

    #Matrix
    def binary_incidence_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import binary_incidence_matrix

        return binary_incidence_matrix(self, return_mapping)

    def incidence_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import incidence_matrix

        return incidence_matrix(self, return_mapping)

    def adjacency_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import adjacency_matrix

        return adjacency_matrix(self, return_mapping)

    #Utils
    def isolated_nodes(self, size=None, order=None):
        from hypergraphx.utils.cc import isolated_nodes

        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.cc import is_isolated

        return is_isolated(self, node, size=size, order=order)

    def dual_random_walk_adjacency(self, return_mapping: bool = False):
        from hypergraphx.linalg import dual_random_walk_adjacency

        return dual_random_walk_adjacency(self, return_mapping)

    def adjacency_factor(self,t:int = 0):
        from hypergraphx.linalg import adjacency_factor
        return adjacency_factor(self, t)

    #Metadata
    def set_hypergraph_metadata(self, metadata):
        self._hypergraph_metadata = metadata

    def get_hypergraph_metadata(self):
        return self._hypergraph_metadata

    def set_node_metadata(self, node, metadata):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self._node_metadata[node]

    def get_all_nodes_metadata(self):
        return self._node_metadata

    def set_edge_metadata(self, edge, metadata):
        edge = tuple(sorted(edge))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._edge_metadata[self._edge_list[edge]] = metadata

    def get_edge_metadata(self, edge):
        edge = tuple(sorted(edge))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self._edge_metadata[self._edge_list[edge]]

    def get_all_edges_metadata(self):
        return self._edge_metadata

    def set_incidence_metadata(self, edge, node, metadata):
        if tuple(sorted(edge)) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._incidences_metadata[(edge, node)] = metadata

    def get_incidence_metadata(self, edge, node):
        if tuple(sorted(edge)) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self._incidences_metadata[(edge, node)]

    def get_all_incidences_metadata(self):
        return {k: v for k, v in self._incidences_metadata.items()}

    def set_attr_to_hypergraph_metadata(self, field, value):
        self._hypergraph_metadata[field] = value

    def set_attr_to_node_metadata(self, node, field, value):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node][field] = value

    def set_attr_to_edge_metadata(self, edge, field, value):
        edge = tuple(sorted(edge))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._edge_metadata[self._edge_list[edge]][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        del self._node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge, field):
        edge = tuple(sorted(edge))
        if edge not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        del self._edge_metadata[self._edge_list[edge]][field]

    #Basic Functions
    def clear(self):
        self._edge_list.clear()
        self._adj.clear()
        self._weights.clear()
        self._hypergraph_metadata.clear()
        self._incidences_metadata.clear()
        self._node_metadata.clear()
        self._edge_metadata.clear()
        self._empty_edges.clear()
        self._reverse_edge_list.clear()

    def copy(self):
        """
        Returns a copy of the hypergraph.

        Returns
        -------
        Hypergraph
            A copy of the hypergraph.
        """
        return copy.deepcopy(self)

    def __str__(self):
        """
        Returns a string representation of the hypergraph.

        Returns
        -------
        str
            A string representation of the hypergraph.
        """
        title = "Hypergraph with {} nodes and {} edges.\n".format(
            self.num_nodes(), self.num_edges()
        )
        details = "Distribution of hyperedge sizes: {}".format(
            self.distribution_sizes()
        )
        return title + details

    def __len__(self):
        """
        Returns the number of edges in the hypergraph.

        Returns
        -------
        int
            The number of edges in the hypergraph.
        """
        return len(self._edge_list)

    def __iter__(self):
        """
        Returns an iterator over the edges in the hypergraph.

        Returns
        -------
        iterator
            An iterator over the edges in the hypergraph.
        """
        return iter(self._edge_list.items())

    #Data Structure Extra
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
                    "metadata": self._edge_metadata.get(edge_id, {}),
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

    def get_mapping(self):
        """
        Map the nodes of the hypergraph to integers in [0, n_nodes).

        Returns
        -------
        LabelEncoder
            The mapping.
        """
        encoder = LabelEncoder()
        encoder.fit(self.get_nodes())
        return encoder
