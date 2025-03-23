import copy
import math

from sklearn.preprocessing import LabelEncoder

from hypergraphx import Hypergraph


def _canon_edge(edge):
    edge = tuple(edge)

    if len(edge) == 2:
        if isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
            # Sort the inner tuples and return
            return (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        elif not isinstance(edge[0], tuple) and not isinstance(edge[1], tuple):
            # Sort the edge itself if it contains IDs (non-tuple elements)
            return tuple(sorted(edge))

    return tuple(sorted(edge))


def _get_size(edge):
    if len(edge) == 2 and isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
        return len(edge[0]) + len(edge[1])
    else:
        return len(edge)


def _get_order(edge):
    return _get_size(edge) - 1


class TemporalHypergraph:
    """
    A Temporal Hypergraph is a hypergraph where each hyperedge is associated with a specific timestamp.
    Temporal hypergraphs are useful for modeling systems where interactions between nodes change over time, such as social networks,
    communication networks, and transportation systems.
    """

    def __init__(
        self,
        edge_list=None,
        time_list=None,
        weighted=False,
        weights=None,
        hypergraph_metadata=None,
        node_metadata=None,
        edge_metadata=None,
    ):
        """
        Initialize a Temporal Hypergraph with optional edges, times, weights, and metadata.

        Parameters
        ----------
        edge_list : list of tuples, optional
            A list of edges where each edge is represented as a tuple of nodes.
            If `time_list` is not provided, each tuple in `edge_list` should
            have the format `(time, edge)`, where `edge` is itself a tuple of nodes.
        time_list : list of int, optional
            A list of times corresponding to each edge in `edge_list`.
            Must be provided if `edge_list` does not include time information.
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
            If `edge_list` and `time_list` have mismatched lengths.
            If `edge_list` contains improperly formatted edges when `time_list` is None.
            If `time_list` is provided without `edge_list`.
        """
        # Initialize hypergraph metadata
        self._hypergraph_metadata = hypergraph_metadata or {}
        self._hypergraph_metadata.update(
            {"weighted": weighted, "type": "TemporalHypergraph"}
        )

        # Initialize core attributes
        self._weighted = weighted
        self._weights = {}
        self._adj = {}
        self._edge_list = {}
        self._incidences_metadata = {}
        self._node_metadata = {}
        self._edge_metadata = {}
        self._reverse_edge_list = {}
        self._next_edge_id = 0

        # Add node metadata if provided
        if node_metadata:
            for node, metadata in node_metadata.items():
                self.add_node(node, metadata=metadata)

        # Handle edge and time list consistency
        if edge_list is not None and time_list is None:
            # Extract times from the edge list if time information is embedded
            if not all(
                isinstance(edge, tuple) and len(edge) == 2 for edge in edge_list
            ):
                raise ValueError(
                    "If time_list is not provided, edge_list must contain tuples of the form (time, edge)."
                )
            time_list = [edge[0] for edge in edge_list]
            edge_list = [edge[1] for edge in edge_list]

        if edge_list is None and time_list is not None:
            raise ValueError("Edge list must be provided if time list is provided.")

        if edge_list is not None and time_list is not None:
            if len(edge_list) != len(time_list):
                raise ValueError("Edge list and time list must have the same length.")
            self.add_edges(
                edge_list, time_list, weights=weights, metadata=edge_metadata
            )

    # Node
    def add_node(self, node, metadata=None):
        if metadata is None:
            metadata = {}
        if node not in self._node_metadata:
            self._adj[node] = []
            self._node_metadata[node] = {}
        if self._node_metadata[node] == {}:
            self._node_metadata[node] = metadata

    def add_nodes(self, node_list: list, metadata=None):
        for node in node_list:
            try:
                self.add_node(node, metadata[node] if metadata is not None else None)
            except KeyError:
                raise ValueError(
                    "The metadata dictionary must contain an entry for each node in the node list."
                )

    def remove_node(self, node, keep_edges=False):
        """
        Remove a node from the temporal hypergraph.

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
                time, edge = self._reverse_edge_list[edge_id]
                updated_edge = tuple(n for n in edge if n != node)

                self.remove_edge((time, edge))
                if updated_edge:
                    self.add_edge(
                        updated_edge,
                        time,
                        weight=self._weights.get(edge_id, 1),
                        metadata=self._edge_metadata.get(edge_id, {}),
                    )
        else:
            for edge_id in edges_to_process:
                time, edge = self._reverse_edge_list[edge_id]
                self.remove_edge((time, edge))

        del self._adj[node]
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
        if metadata:
            return self._node_metadata
        return list(self._node_metadata.keys())

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
                neigh.update(edge[1])
            if node in neigh:
                neigh.remove(node)
            return neigh
        else:
            if order is None:
                order = size - 1
            neigh = set()
            edges = self.get_incident_edges(node, order=order)
            for edge in edges:
                neigh.update(edge[1])
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
                    if len(self._reverse_edge_list[edge_id][1]) - 1 == order
                ]
            )

    # Edge
    def add_edge(self, edge, time, weight=None, metadata=None):
        """
        Add an edge to the temporal hypergraph. If the edge already exists, the weight is updated.

        Parameters
        ----------
        edge : tuple
            The edge to add. If the hypergraph is undirected, should be a tuple.
            If the hypergraph is directed, should be a tuple of two tuples.
        time: int
            The time at which the edge occurs.
        weight: float, optional
            The weight of the edge. Default is None.

        metadata: dict, optional
            Metadata for the edge. Default is an empty dictionary.

        Raises
        ------
        TypeError
            If time is not an integer.
        ValueError
            If the hypergraph is not weighted and weight is not None or 1.
        """
        if not isinstance(time, int):
            raise TypeError("Time must be an integer")

        if not self._weighted and weight is not None and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )
        if weight is None:
            weight = 1

        t = time

        if t < 0:
            raise ValueError("Time must be a positive integer")

        _edge = _canon_edge(edge)
        edge = (t, _edge)

        if edge not in self._edge_list:
            e_id = self._next_edge_id
            self._reverse_edge_list[e_id] = edge
            self._edge_list[edge] = e_id
            self._next_edge_id += 1
            self._weights[e_id] = weight
        elif edge in self._edge_list and self._weighted:
            self._weights[self._edge_list[edge]] += weight

        e_id = self._edge_list[edge]

        if metadata is None:
            metadata = {}
        self._edge_metadata[e_id] = metadata

        for node in _edge:
            self.add_node(node)

        for node in _edge:
            self._adj[node].append(e_id)

    def add_edges(self, edge_list, time_list, weights=None, metadata=None):
        """
        Add multiple edges to the temporal hypergraph.

        Parameters
        ----------
        edge_list: list
            A list of edges to add.
        time_list: list
            A list of times corresponding to each edge in `edge_list`.
        weights: list, optional
            A list of weights for each edge in `edge_list`. Must be provided if the hypergraph is weighted.
        metadata: list, optional
            A list of metadata dictionaries for each edge in `edge_list`.

        Raises
        ------
        TypeError
            If `edge_list` and `time_list` are not lists.
        ValueError
            If `edge_list` and `time_list` have mismatched lengths.
        """
        if not isinstance(edge_list, list) or not isinstance(time_list, list):
            raise TypeError("Edge list and time list must be lists")

        if len(edge_list) != len(time_list):
            raise ValueError("Edge list and time list must have the same length")

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
                    time_list[i],
                    weight=(
                        weights[i] if self._weighted and weights is not None else None
                    ),
                    metadata=metadata[i] if metadata is not None else None,
                )
                i += 1

    def remove_edge(self, edge, time):
        """
        Remove an edge from the temporal hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to remove.
        time : int
            The time at which the edge occurs.

        Raises
        ------
        ValueError
            If the edge is not in the hypergraph.
        """
        edge = _canon_edge(edge)
        edge = (time, edge)
        if edge not in self._edge_list:
            raise ValueError(f"Edge {edge} not in hypergraph.")
        edge_id = self._edge_list[edge]

        # Remove edge from reverse lookup and metadata
        del self._reverse_edge_list[edge_id]
        if edge_id in self._weights:
            del self._weights[edge_id]
        if edge_id in self._edge_metadata:
            del self._edge_metadata[edge_id]

        time, nodes = edge
        for node in nodes:
            if edge_id in self._adj[node]:
                self._adj[node].remove(edge_id)

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

    def get_edge_list(self):
        return self._edge_list

    def set_edge_list(self, edge_list):
        self._edge_list = edge_list

    def check_edge(self, edge, time):
        """Checks if the specified edge is in the hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to check.
        time : int
            The time to check.
        Returns
        -------
        bool
            True if the edge is in the hypergraph, False otherwise.

        """
        edge = _canon_edge(edge)
        k = (time, edge)
        return k in self._edge_list

    def get_edges(
        self,
        time_window=None,
        order=None,
        size=None,
        up_to=False,
        # subhypergraph = False,
        # keep_isolated_nodes=False,
        metadata=False,
    ):
        """
        Get the edges in the temporal hypergraph. If a time window is provided, only edges within the window are returned.

        Parameters
        ----------
        time_window: tuple, optional
            A tuple of two integers representing the start and end times of the window.
        size: int, optional
            The size of the hyperedges to consider
        order: int, optional
            The order of the hyperedges to consider
        up_to: bool, optional
        metadata: bool, optional
            If True, return edge metadata. Default is False.

        Returns
        -------
        list
            A list of edges in the hypergraph.
        """
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        # if not subhypergraph and keep_isolated_nodes:
        #    raise ValueError("Cannot keep nodes if not returning subhypergraphs.")

        edges = []
        if time_window is None:
            edges = list(self._edge_list.keys())
        elif isinstance(time_window, tuple) and len(time_window) == 2:
            for _t, _edge in list(sorted(self._edge_list.keys())):
                if time_window[0] <= _t < time_window[1]:
                    edges.append((_t, _edge))
        else:
            raise ValueError("Time window must be a tuple of length 2 or None")
        if order is not None or size is not None:
            if size is not None:
                order = size - 1
            if not up_to:
                edges = [edge for edge in edges if len(edge[1]) - 1 == order]
            else:
                edges = [edge for edge in edges if len(edge[1]) - 1 <= order]
        return (
            edges
            if not metadata
            else {edge: self.get_edge_metadata(edge[1], edge[0]) for edge in edges}
        )

    def aggregate(self, time_window):
        if not isinstance(time_window, int) or time_window <= 0:
            raise TypeError("Time window must be a positive integer")

        aggregated = {}
        node_list = self.get_nodes()

        # Get all edges and determine the max time
        sorted_edges = sorted(self.get_edges())
        if not sorted_edges:
            return aggregated  # Return empty if no edges exist

        max_time = max(edge[0] for edge in sorted_edges)  # Maximum time of all edges

        # Initialize time window boundaries
        t_start = 0
        t_end = time_window
        edges_in_window = []
        num_windows_created = 0

        edge_index = 0  # Pointer to the current edge in sorted_edges

        while t_start <= max_time:
            # Collect edges for the current window
            while (
                edge_index < len(sorted_edges)
                and t_start <= sorted_edges[edge_index][0] < t_end
            ):
                edges_in_window.append(sorted_edges[edge_index])
                edge_index += 1

            # Create the hypergraph for this time window
            Hypergraph_t = Hypergraph(weighted=self._weighted)

            # Add edges to the hypergraph
            for time, edge_nodes in edges_in_window:
                Hypergraph_t.add_edge(
                    edge_nodes,
                    metadata=self.get_edge_metadata(edge_nodes, time),
                    weight=self.get_weight(edge_nodes, time),
                )

            # Add all nodes to ensure node consistency
            for node in node_list:
                Hypergraph_t.add_node(node, metadata=self._node_metadata[node])

            # Store the finalized hypergraph for this window
            aggregated[num_windows_created] = Hypergraph_t
            num_windows_created += 1

            # Advance to the next time window
            t_start = t_end
            t_end += time_window
            edges_in_window = []  # Reset for the next window

        return aggregated

    def get_times_for_edge(self, edge):
        """
        Get the times at which a specific set of nodes forms a hyperedge in the hypergraph.

        Parameters
        ----------
        edge: tuple
            The set of nodes forming the hyperedge.

        Returns
        -------
        times: list
            A list of times at which the hyperedge occurs.
        """
        edge = _canon_edge(edge)
        times = []
        for time, _edge in self._edge_list.keys():
            if _edge == edge:
                times.append(time)
        return times

    # Weight
    def set_weight(self, edge, time, weight):
        edge = _canon_edge(edge)
        if not self._weighted and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )
        if (time, edge) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[(time, edge)]
        self._weights[e_id] = weight

    def get_weight(self, edge, time):
        edge = _canon_edge(edge)
        if (time, edge) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[(time, edge)]
        return self._weights[e_id]

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

        asdict : bool, optional
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

    # Info
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

    def get_sizes(self):
        """
        Get the size of each edge in the hypergraph.

        Returns
        -------
        list
            A list of integers representing the size of each edge.
        """
        return [_get_size(edge[1]) for edge in self._edge_list.keys()]

    def get_orders(self):
        """
        Get the order of each edge in the hypergraph.

        Returns
        -------
        list
            A list of integers representing the order of each edge.
        """
        return [_get_order(edge[1]) for edge in self._edge_list.keys()]

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
                sz = len(edge[1])
            else:
                if len(edge[1]) != sz:
                    uniform = False
                    break
        return uniform

    def min_time(self):
        min = math.inf
        for edge in self._edge_list:
            if min > edge[0]:
                min = edge[0]
        return min

    def max_time(self):
        max = -math.inf
        for edge in self._edge_list:
            if max < edge[0]:
                max = edge[0]
        return max

    # Adj
    def get_adj_dict(self):
        return self._adj

    def set_adj_dict(self, adj_dict):
        self._adj = adj_dict

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

    # Utils
    def isolated_nodes(self, size=None, order=None):
        from hypergraphx.utils.cc import isolated_nodes

        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.cc import is_isolated

        return is_isolated(self, node, size=size, order=order)

    def temporal_adjacency_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import temporal_adjacency_matrix

        return temporal_adjacency_matrix(self, return_mapping)

    def annealed_adjacency_matrix(self, return_mapping: bool = False):
        from hypergraphx.linalg import annealed_adjacency_matrix

        return annealed_adjacency_matrix(self, return_mapping)

    def adjacency_factor(self, t: int = 0):
        from hypergraphx.linalg import adjacency_factor

        return adjacency_factor(self, t)

    def subhypergraph(
        self, time_window=None, add_all_nodes: bool = False
    ) -> dict[int, Hypergraph]:
        """
        Create an hypergraph for each time of the Temporal Hypergraph.
        Parameters
        ----------
        time_window : tuple[int,int]|None, optional
            Give the time window (a,b), only the times inside the interval [a,b) will be considered.
            If not specified all the times will be considered.
        add_all_nodes : bool, optional
            If True, the hypergraphs will have all the nodes of the Temporal Hypergraph even if they are not present
            in their corresponding time.
        Returns
        -------
        dict: dict[int, Hypergraph]
            A dictionary where the keys are the time and the values are the hypergraphs
        """
        edges = self.get_edges()
        res = dict()
        if time_window is None:
            time_window = (-math.inf, math.inf)
        if not isinstance(time_window, tuple):
            raise ValueError("Time window must be a tuple of length 2 or None")

        for edge in edges:
            if time_window[0] <= edge[0] < time_window[1]:
                if edge[0] not in res.keys():
                    res[edge[0]] = Hypergraph(weighted=self.is_weighted())
                weight = self.get_weight(edge[1], edge[0])
                res[edge[0]].add_edge(edge[1], weight)
        if add_all_nodes:
            for node in self.get_nodes():
                for k, v in res.items():
                    if v.check_node(node):
                        v.add_node(node)

        return res

    # Metadata
    def set_hypergraph_metadata(self, metadata):
        self._hypergraph_metadata = metadata

    def get_hypergraph_metadata(self):
        return self._hypergraph_metadata

    def set_node_metadata(self, node, metadata):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self._node_metadata[node]

    def get_all_nodes_metadata(self):
        return self._node_metadata

    def set_edge_metadata(self, edge, time, metadata):
        edge = _canon_edge(edge)
        k = (time, edge)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[k]
        self._edge_metadata[e_id] = metadata

    def get_edge_metadata(self, edge, time):
        edge = _canon_edge(edge)
        k = (time, edge)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[k]
        return self._edge_metadata[e_id]

    def get_all_edges_metadata(self):
        return self._edge_metadata

    def set_incidence_metadata(self, edge, time, node, metadata):
        edge = _canon_edge(edge)
        k = (time, edge)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._incidences_metadata[(k, node)] = metadata

    def get_incidence_metadata(self, edge, time, node):
        edge = _canon_edge(edge)
        k = (time, edge)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self._incidences_metadata[(k, node)]

    def get_all_incidences_metadata(self):
        return {k: v for k, v in self._incidences_metadata.items()}

    def set_attr_to_hypergraph_metadata(self, field, value):
        self._hypergraph_metadata[field] = value

    def set_attr_to_node_metadata(self, node, field, value):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node][field] = value

    def set_attr_to_edge_metadata(self, edge, time, field, value):
        _edge = _canon_edge(edge)
        if edge not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._edge_metadata[self._edge_list[(time, edge)]][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        del self._node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge, time, field):
        _edge = _canon_edge(edge)
        if edge not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        del self._edge_metadata[self._edge_list[(time, edge)]][field]

    # Basic Functions
    def clear(self):
        self._edge_list.clear()
        self._adj.clear()
        self._weights.clear()
        self._hypergraph_metadata.clear()
        self._node_metadata.clear()
        self._edge_metadata.clear()
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

    # Data Structure Extra
    def expose_data_structures(self):
        """
        Expose the internal data structures of the temporal hypergraph for serialization.

        Returns
        -------
        dict
            A dictionary containing all internal attributes of the temporal hypergraph.
        """
        return {
            "type": "TemporalHypergraph",
            "hypergraph_metadata": self._hypergraph_metadata,
            "_weighted": self._weighted,
            "_weights": self._weights,
            "_adj": self._adj,
            "_edge_list": self._edge_list,
            "node_metadata": self._node_metadata,
            "edge_metadata": self._edge_metadata,
            "reverse_edge_list": self._reverse_edge_list,
            "next_edge_id": self._next_edge_id,
        }

    def populate_from_dict(self, data):
        """
        Populate the attributes of the temporal hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        self._hypergraph_metadata = data.get("hypergraph_metadata", {})
        self._weighted = data.get("_weighted", False)
        self._weights = data.get("_weights", {})
        self._adj = data.get("_adj", {})
        self._edge_list = data.get("_edge_list", {})
        self._node_metadata = data.get("node_metadata", {})
        self._edge_metadata = data.get("edge_metadata", {})
        self._reverse_edge_list = data.get("reverse_edge_list", {})
        self._next_edge_id = data.get("next_edge_id", 0)

    def expose_attributes_for_hashing(self):
        """
        Expose relevant attributes for hashing specific to TemporalHypergraph.

        Returns
        -------
        dict
            A dictionary containing key attributes.
        """
        edges = []
        for edge in sorted(self._edge_list.keys()):
            edge = (edge[0], tuple(sorted(edge[1])))
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
            "type": "TemporalHypergraph",
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
