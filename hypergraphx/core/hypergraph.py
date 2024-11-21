import copy
from typing import Optional, Tuple
from sklearn.preprocessing import LabelEncoder

class Hypergraph:
    """
    A hypergraph is a set of nodes and a set of hyperedges, where each hyperedge is a subset of the nodes.
    Hypergraphs are represented as a dictionary with keys being tuples of nodes (hyperedges) and values being the weights
    of the hyperedges (if the hypergraph is weighted).

    Parameters
    ----------
    edge_list : list of tuples, optional
        A list of tuples representing the hyperedges of the hypergraph.
    weighted : bool
        Whether the hypergraph is weighted.
    weights : list of floats, optional
        A list of weights for the hyperedges. If the hypergraph is weighted, this must be provided.
    """

    def __init__(self, edge_list=None, hypergraph_metadata=None, weighted=False, weights=None, edge_metadata=None):
        self._weighted = weighted
        self._adj = {}
        self._edge_list = {}
        if hypergraph_metadata is None:
            self.hypergraph_metadata = {}
        else:
            self.hypergraph_metadata = hypergraph_metadata
        self.hypergraph_metadata['weighted'] = weighted
        self.incidences_metadata = {}
        self.node_metadata = {}
        self.edge_metadata = {}
        self.empty_edges = {}

        if edge_list is not None:
            self.add_edges(edge_list, weights=weights, metadata=edge_metadata)

    def add_empty_edge(self, name, metadata):
        if name not in self.empty_edges:
            self.empty_edges[name] = metadata
        else:
            raise("Edge {} already in hypergraph.".format(name))

    def get_hypergraph_metadata(self):
        return self.hypergraph_metadata

    def set_hypergraph_metadata(self, metadata):
        self.hypergraph_metadata = metadata

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
        """
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return list(self._adj[node])
        else:
            if order is None:
                order = size - 1
            return list([edge for edge in self._adj[node] if len(edge) - 1 == order])

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
        if node not in self._adj:
            self._adj[node] = set()
            if metadata is not None:
                self.node_metadata[node] = metadata
            else:
                self.node_metadata[node] = {}

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
        for node in node_list:
            try:
                self.add_node(node, node_metadata[node] if node_metadata is not None else None)
            except KeyError:
                raise ValueError("The metadata dictionary must contain an entry for each node in the node list.")

    def is_weighted(self):
        """
        Check if the hypergraph is weighted.

        Returns
        -------
        bool
            True if the hypergraph is weighted, False otherwise.
        """
        return self._weighted

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
        if self._weighted and weight is None:
            raise ValueError(
                "If the hypergraph is weighted, a weight must be provided."
            )
        if not self._weighted and weight is not None:
            raise ValueError(
                "If the hypergraph is not weighted, no weight must be provided."
            )

        edge = tuple(sorted(edge))
        order = len(edge) - 1
        if metadata is None:
            metadata = {}
        self.edge_metadata[edge] = metadata

        if weight is None:
            if edge in self._edge_list and self._weighted:
                self._edge_list[edge] += 1
            else:
                self._edge_list[edge] = 1
        else:
            self._edge_list[edge] = weight

        for node in edge:
            self.add_node(node)
            self._adj[node].add(edge)

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
                    weight=weights[i]
                    if self._weighted and weights is not None
                    else None,
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
        try:
            del self._edge_list[edge]
            order = len(edge) - 1
            for node in edge:
                self._adj[node].remove(edge)
        except KeyError:
            print("Edge {} not in hypergraph.".format(edge))

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
            self.remove_edges(list(self._adj[node]))
        else:
            for edge in self._adj[node]:
                self.add_edge(
                    tuple(sorted([n for n in edge if n != node])),
                    weight=self.get_weight(edge),
                    metadata=self.get_edge_metadata(edge),
                )
                self.remove_edge(edge)
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
                    h.add_edge(edge, weight=self._edge_list[edge], metadata=self.get_edge_metadata(edge))
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
                    h.add_edge(edge, self.get_weight(edge), self.get_edge_metadata(edge))
                else:
                    h.add_edge(edge, metadata=self.get_edge_metadata(edge))

        return h

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
        try:
            return self._edge_list[tuple(sorted(edge))]
        except KeyError:
            raise ValueError("Edge {} not in hypergraph.".format(edge))

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
        try:
            self._edge_list[tuple(sorted(edge))] = weight
        except KeyError:
            raise ValueError("Edge {} not in hypergraph.".format(edge))

    def get_weights(self, order=None, size=None, up_to=False):
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
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return {edge: self._edge_list[edge] for edge in self.get_edges()}

        if size is not None:
            order = size - 1
        return {edge: self._edge_list[edge] for edge in self.get_edges(order=order, up_to=up_to)}

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

    def set_node_metadata(self, node, metadata):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self.node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self.node_metadata[node]

    def get_all_nodes_metadata(self):
        return list(self.node_metadata.values())

    def set_edge_metadata(self, edge, metadata):
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self.edge_metadata[edge] = metadata

    def get_edge_metadata(self, edge):
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self.edge_metadata[edge]

    def set_incidence_metadata(self, edge, node, metadata):
        if tuple(sorted(edge)) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self.incidences_metadata[(edge, node)] = metadata

    def get_incidence_metadata(self, edge, node):
        if tuple(sorted(edge)) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self.incidences_metadata[(edge, node)]

    def get_all_incidences_metadata(self):
        return { k: v for k, v in self.incidences_metadata.items() }
    
    def get_hypergraph_metadata(self):
        return self.hypergraph_metadata
    
    def set_hypergraph_metadata(self, metadata):
        self.hypergraph_metadata = metadata

    def get_node_metadata(self, node):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self.node_metadata[node]
    
    def set_node_metadata(self, node, metadata):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self.node_metadata[node] = metadata

    def get_all_nodes_metadata(self):
        return { k: v for k, v in self.node_metadata.items() }
    
    def get_edge_metadata(self, edge):
        edge = tuple(sorted(edge))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self.edge_metadata[edge]
    
    def set_edge_metadata(self, edge, metadata):
        edge = tuple(sorted(edge))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self.edge_metadata[edge] = metadata

    def get_all_edges_metadata(self):
        return { k: v for k, v in self.edge_metadata.items() }

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

    def get_edges(
        self,
        order=None,
        size=None,
        up_to=False,
        subhypergraph=False,
        keep_isolated_nodes=False,
        metadata=False
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
                edges = [edge for edge in list(self._edge_list.keys()) if len(edge) - 1 == order]
            else:
                edges = [edge for edge in list(self._edge_list.keys()) if len(edge) - 1 <= order]

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
            return edges if not metadata else {edge: self.get_edge_metadata(edge) for edge in edges}

    def degree(self, node, order=None, size=None):
        from hypergraphx.measures.degree import degree

        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_sequence

        return degree_sequence(self, order=order, size=size)

    def degree_distribution(self, order=None, size=None):
        from hypergraphx.measures.degree import degree_distribution

        return degree_distribution(self, order=order, size=size)

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

    def isolated_nodes(self, size=None, order=None):
        from hypergraphx.utils.cc import isolated_nodes

        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.cc import is_isolated

        return is_isolated(self, node, size=size, order=order)

    def binary_incidence_matrix(
        self, return_mapping: bool = False
    ):
        from hypergraphx.linalg import binary_incidence_matrix
        return binary_incidence_matrix(self, return_mapping)

    def incidence_matrix(
        self, return_mapping: bool = False
    ):
        from hypergraphx.linalg import incidence_matrix

        return incidence_matrix(self, return_mapping)

    def adjacency_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import adjacency_matrix

        return adjacency_matrix(self, return_mapping)

    def dual_random_walk_adjacency(self, return_mapping: bool = False):
        from hypergraphx.linalg import dual_random_walk_adjacency

        return dual_random_walk_adjacency(self, return_mapping)

    def clear(self):
        self._edge_list.clear()

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
