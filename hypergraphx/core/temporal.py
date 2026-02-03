import math
from .undirected import Hypergraph
from hypergraphx.core.base import BaseHypergraph
from hypergraphx.exceptions import InvalidParameterError, MissingNodeError
from hypergraphx.utils.edges import canon_edge


def _get_size(edge):
    if len(edge) == 2 and isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
        return len(edge[0]) + len(edge[1])
    else:
        return len(edge)


def _get_order(edge):
    return _get_size(edge) - 1


def _get_nodes(edge):
    if len(edge) == 2 and isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
        return list(edge[0]) + list(edge[1])
    else:
        return list(edge)


class TemporalHypergraph(BaseHypergraph):
    """
    A Temporal Hypergraph is a hypergraph where each hyperedge is associated with a specific timestamp.
    Temporal hypergraphs are useful for modeling systems where interactions between nodes change over time, such as social networks,
    communication networks, and transportation systems.
    """

    def __init__(
        self,
        edge_list=None,
        time_list=None,
        weighted=True,
        weights=None,
        hypergraph_metadata=None,
        node_metadata=None,
        edge_metadata=None,
        duplicate_policy=None,
        metadata_policy=None,
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
            If `edge_list` and `time_list` have mismatched lengths.
            If `edge_list` contains improperly formatted edges when `time_list` is None.
            If `time_list` is provided without `edge_list`.
        """
        self._adj = {}
        metadata = hypergraph_metadata or {}
        metadata.update({"weighted": weighted, "type": "TemporalHypergraph"})
        self._init_base(
            weighted=weighted,
            hypergraph_metadata=metadata,
            node_metadata=node_metadata,
            duplicate_policy=duplicate_policy,
            metadata_policy=metadata_policy,
        )

        # Handle edge and time list consistency
        if edge_list is not None and time_list is None:
            # Extract times from the edge list if time information is embedded
            if not all(
                isinstance(e, tuple)
                and len(e) == 2
                and isinstance(e[0], int)
                and isinstance(e[1], (tuple, list))
                for e in edge_list
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

    def _normalize_edge(self, edge, time=None, **kwargs):
        if time is None:
            if isinstance(edge, tuple) and len(edge) == 2:
                if isinstance(edge[0], int) and isinstance(edge[1], (tuple, list)):
                    time = edge[0]
                    edge = edge[1]
                elif isinstance(edge[1], int) and isinstance(edge[0], (tuple, list)):
                    time = edge[1]
                    edge = edge[0]
                else:
                    raise ValueError(
                        "Temporal edges must be provided as (time, edge) or with a time argument."
                    )
            else:
                raise ValueError(
                    "Temporal edges must be provided as (time, edge) or with a time argument."
                )
        return (time, canon_edge(edge))

    def _edge_nodes(self, edge_key):
        return _get_nodes(edge_key[1])

    def _edge_size(self, edge_key):
        return _get_size(edge_key[1])

    def _edge_key_without_node(self, edge_key, node):
        time, edge = edge_key
        return (time, tuple(n for n in edge if n != node))

    def _allow_empty_edge(self):
        return False

    def _new_like(self):
        return TemporalHypergraph(weighted=self._weighted)

    def _hash_edge_nodes(self, edge_key):
        return (edge_key[0], tuple(sorted(edge_key[1])))

    # Node
    def add_node(self, node, metadata=None):
        super().add_node(node, metadata=metadata)

    def add_nodes(self, node_list: list, metadata=None):
        super().add_nodes(node_list, metadata=metadata)

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
        super().remove_node(node, keep_edges=keep_edges)

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
        super().remove_nodes(node_list, keep_edges=keep_edges)

    def get_nodes(self, metadata=False):
        return super().get_nodes(metadata=metadata)

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
        return super().check_node(node)

    def get_neighbors(self, node, order: int = None, size: int = None):
        return super().get_neighbors(node, order=order, size=size)

    def get_incident_edges(self, node, order: int = None, size: int = None):
        return super().get_incident_edges(node, order=order, size=size)

    # Edge
    def add_edge(
        self,
        edge,
        time=None,
        weight=None,
        metadata=None,
    ):
        """
        Add an edge to the temporal hypergraph. If the edge already exists, the weight is updated.

        Parameters
        ----------
        edge : tuple
            The edge to add, or a packed temporal edge key `(time, edge)`.
            If the hypergraph is undirected, `edge` should be a tuple of nodes.
            If the hypergraph is directed, `edge` should be a tuple of two tuples.
        time: int, optional
            The time at which the edge occurs. If not provided, `edge` must be a packed
            `(time, edge)` tuple (or `(edge, time)`).
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
        Notes
        -----
        Duplicate unweighted edges are ignored; duplicate weighted edges accumulate weights.
        """
        edge_key = self._normalize_edge(edge, time=time)
        time = edge_key[0]
        if not isinstance(time, int):
            raise TypeError("Time must be an integer.")
        if time < 0:
            raise ValueError("Time must be a non-negative integer.")
        self._add_edge(edge_key, weight=weight, metadata=metadata)

    def add_edges(
        self,
        edge_list,
        time_list=None,
        weights=None,
        metadata=None,
    ):
        """
        Add multiple edges to the temporal hypergraph.

        Parameters
        ----------
        edge_list: iterable
            An iterable of edges to add. If `time_list` is not provided, it must contain
            packed `(time, edge)` tuples.
        time_list: iterable, optional
            An iterable of times corresponding to each edge in `edge_list`.
        weights: list, optional
            A list of weights for each edge in `edge_list`. Must be provided if the hypergraph is weighted.
        metadata: list, optional
            A list of metadata dictionaries for each edge in `edge_list`.

        Raises
        ------
        TypeError
            If `edge_list` and `time_list` are not iterable.
        ValueError
            If `edge_list` and `time_list` have mismatched lengths.
        """
        try:
            edge_list = list(edge_list)
            if time_list is not None:
                time_list = list(time_list)
        except TypeError as exc:
            raise TypeError("Edge list and time list must be iterable") from exc
        if weights is not None:
            weights = list(weights)
        if metadata is not None:
            metadata = list(metadata)

        if time_list is None:
            if not all(
                isinstance(e, tuple)
                and len(e) == 2
                and isinstance(e[0], int)
                and isinstance(e[1], (tuple, list))
                for e in edge_list
            ):
                raise ValueError(
                    "If time_list is not provided, edge_list must contain tuples of the form (time, edge)."
                )
            time_list = [e[0] for e in edge_list]
            edge_list = [e[1] for e in edge_list]

        if len(edge_list) != len(time_list):
            raise ValueError("Edge list and time list must have the same length")

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
                    time_list[i],
                    weight=(
                        weights[i] if self._weighted and weights is not None else None
                    ),
                    metadata=metadata[i] if metadata is not None else None,
                )

    def remove_edge(self, edge, time=None):
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
        edge_key = self._normalize_edge(edge, time=time)
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

    def get_edge_list(self):
        return self._edge_list

    def set_edge_list(self, edge_list):
        self._guard_unsafe_setter("TemporalHypergraph.set_edge_list")
        self._edge_list = edge_list
        self._maybe_validate_invariants()

    def check_edge(self, edge, time=None):
        """Checks if the specified edge is in the hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to check.
        time : int, optional
            The time to check. If not provided, `edge` must be a packed `(time, edge)` tuple.
        Returns
        -------
        bool
            True if the edge is in the hypergraph, False otherwise.

        """
        edge_key = self._normalize_edge(edge, time=time)
        return self._edge_exists(edge_key)

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
            raise InvalidParameterError("Order and size cannot be both specified.")
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
        edges = self._filter_edges_by_order(edges, order=order, size=size, up_to=up_to)
        return (
            edges
            if not metadata
            else {edge: self.get_edge_metadata(edge) for edge in edges}
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
        edge = canon_edge(edge)
        times = []
        for time, _edge in self._edge_list.keys():
            if _edge == edge:
                times.append(time)
        return times

    # Weight
    def set_weight(self, edge, time=None, weight=None):
        # Support set_weight((time, edge), weight) with positional weight
        if weight is None and time is not None:
            if (
                isinstance(edge, tuple)
                and len(edge) == 2
                and isinstance(edge[0], int)
                and isinstance(edge[1], (tuple, list))
            ):
                weight = time
                time = None
        if weight is None:
            raise TypeError("set_weight() missing required argument: 'weight'")
        edge_key = self._normalize_edge(edge, time=time)
        super().set_weight(edge_key, weight)

    def get_weight(self, edge, time=None):
        edge_key = self._normalize_edge(edge, time=time)
        return super().get_weight(edge_key)

    # Info
    def get_times(self):
        """
        Get the times of each edge in the hypergraph.

        Returns
        -------
        list
            A list of integers representing the times of each edge.
        """
        return [edge[0] for edge in self._edge_list.keys()]

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
            elif len(edge[1]) != sz:
                uniform = False
                break
        return uniform

    def min_time(self):
        min_time = math.inf
        for edge in self._edge_list:
            if min_time > edge[0]:
                min_time = edge[0]
        return min_time

    def max_time(self):
        max_time = -math.inf
        for edge in self._edge_list:
            if max_time < edge[0]:
                max_time = edge[0]
        return max_time

    # Adj
    def get_adj_dict(self):
        return self._adj

    def set_adj_dict(self, adj_dict):
        self._guard_unsafe_setter("TemporalHypergraph.set_adj_dict")
        self._adj = adj_dict
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

    # Utils
    def isolated_nodes(self, size=None, order=None):
        from hypergraphx.utils.components import isolated_nodes

        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hypergraphx.utils.components import is_isolated

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

    def to_hypergraph(
        self,
        keep_node_metadata: bool = True,
        keep_edge_metadata: bool = True,
        keep_hypergraph_metadata: bool = True,
    ):
        """Convert to an undirected Hypergraph by dropping time information.

        Duplicate hyperedges are merged by summing weights and merging metadata.
        """
        from hypergraphx.core.undirected import Hypergraph
        from hypergraphx.utils.metadata import merge_metadata

        hg = Hypergraph(weighted=True)
        if keep_hypergraph_metadata:
            meta = merge_metadata(
                self.get_hypergraph_metadata(), {"converted_from": "TemporalHypergraph"}
            )
            hg.set_hypergraph_metadata(meta)

        if keep_node_metadata:
            for node, metadata in self.get_all_nodes_metadata().items():
                hg.add_node(node, metadata=metadata)

        edge_weights = {}
        edge_metadata = {}
        for time, edge in self.get_edges():
            edge_weights[edge] = edge_weights.get(edge, 0) + self.get_weight(edge, time)
            if keep_edge_metadata:
                edge_metadata[edge] = merge_metadata(
                    edge_metadata.get(edge), self.get_edge_metadata(edge, time)
                )

        for edge, weight in edge_weights.items():
            hg.add_edge(edge, weight=weight, metadata=edge_metadata.get(edge))

        return hg

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
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        self._node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._node_metadata:
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        return self._node_metadata[node]

    def get_all_nodes_metadata(self):
        return self._node_metadata

    def set_edge_metadata(self, edge, time=None, metadata=None):
        # Support set_edge_metadata((time, edge), metadata) with positional metadata
        if metadata is None and isinstance(time, dict):
            metadata = time
            time = None
        edge_key = self._normalize_edge(edge, time=time)
        super().set_edge_metadata(edge_key, metadata)

    def get_edge_metadata(self, edge, time=None):
        edge_key = self._normalize_edge(edge, time=time)
        return super().get_edge_metadata(edge_key)

    def get_all_edges_metadata(self):
        return self._edge_metadata

    def set_incidence_metadata(self, edge, time, node, metadata):
        edge_key = self._normalize_edge(edge, time=time)
        super().set_incidence_metadata(edge_key, node, metadata)

    def get_incidence_metadata(self, edge, time, node):
        edge_key = self._normalize_edge(edge, time=time)
        return super().get_incidence_metadata(edge_key, node)

    def get_all_incidences_metadata(self):
        return {k: v for k, v in self._incidences_metadata.items()}

    def set_attr_to_hypergraph_metadata(self, field, value):
        self._hypergraph_metadata[field] = value

    def set_attr_to_node_metadata(self, node, field, value):
        if node not in self._node_metadata:
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        self._node_metadata[node][field] = value

    def set_attr_to_edge_metadata(self, edge, time, field, value):
        edge_key = self._normalize_edge(edge, time=time)
        super().set_attr_to_edge_metadata(edge_key, field, value)

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self._node_metadata:
            raise MissingNodeError(f"Node {node} not in hypergraph.")
        del self._node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge, time, field):
        edge_key = self._normalize_edge(edge, time=time)
        super().remove_attr_from_edge_metadata(edge_key, field)

    def __repr__(self):
        if not self._edge_list:
            time_info = "times=0"
        else:
            time_info = "time_range=[{}, {}]".format(self.min_time(), self.max_time())
        return "{}(nodes={}, edges={}, {}, weighted={})".format(
            self._type_name(),
            self.num_nodes(),
            self.num_edges(),
            time_info,
            self._weighted,
        )

    def summary(
        self, *, include_size_distribution: bool = True, max_size_bins: int = 20
    ):
        base = super().summary(
            include_size_distribution=include_size_distribution,
            max_size_bins=max_size_bins,
        )
        times = self.get_times()
        base["num_times"] = len(set(times)) if times else 0
        base["min_time"] = self.min_time() if times else None
        base["max_time"] = self.max_time() if times else None
        return base

    # Basic Functions
    def clear(self):
        super().clear()

    # Data Structure Extra
    def populate_from_dict(self, data):
        """
        Populate the attributes of the temporal hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        super().populate_from_dict(data)
