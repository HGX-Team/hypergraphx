from hypergraphx import Hypergraph


class TemporalHypergraph:
    def __init__(
        self,
        edge_list=None,
        time_list=None,
        weighted=False,
        weights=None,
        hypergraph_metadata=None,
        edge_metadata=None,
    ):
        if hypergraph_metadata is None:
            self.hypergraph_metadata = {}
        else:
            self.hypergraph_metadata = hypergraph_metadata
        self.hypergraph_metadata["weighted"] = weighted
        self._weighted = weighted
        self._weights = {}
        self._adj = {}
        self._edge_list = {}
        self.node_metadata = {}
        self.edge_metadata = {}
        self.reverse_edge_list = {}
        self.next_edge_id = 0

        if edge_list is not None and time_list is None:
            if not all(
                isinstance(edge, tuple) and len(edge) == 2 for edge in edge_list
            ):
                raise ValueError(
                    "Edge list must be a list of tuples of two elements if time list is not provided"
                )
            time_list = [edge[0] for edge in edge_list]
            edge_list = [edge[1] for edge in edge_list]
            self.add_edges(
                edge_list, time_list, weights=weights, metadata=edge_metadata
            )
        if edge_list is None and time_list is not None:
            raise ValueError("Edge list must be provided if time list is provided")
        if edge_list is not None and time_list is not None:
            self.add_edges(
                edge_list, time_list, weights=weights, metadata=edge_metadata
            )

    def get_adj_dict(self):
        return self._adj

    def set_adj_dict(self, adj_dict):
        self._adj = adj_dict

    def get_incident_edges(self, node):
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return [self.reverse_edge_list[e_id] for e_id in self._adj[node]]

    def get_edge_list(self):
        return self._edge_list

    def set_edge_list(self, edge_list):
        self._edge_list = edge_list

    def get_hypergraph_metadata(self):
        return self.hypergraph_metadata

    def set_hypergraph_metadata(self, metadata):
        self.hypergraph_metadata = metadata

    def set_node_metadata(self, node, metadata):
        if node not in self.node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self.node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self.node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self.node_metadata[node]

    def get_all_nodes_metadata(self):
        return self.node_metadata

    def set_edge_metadata(self, edge, metadata):
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[edge]
        self.edge_metadata[e_id] = metadata

    def get_edge_metadata(self, edge, time):
        k = (time, edge)
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[k]
        return self.edge_metadata[e_id]

    def get_all_edges_metadata(self):
        return self.edge_metadata

    def add_node(self, node, metadata=None):
        if metadata is None:
            metadata = {}
        if node not in self.node_metadata:
            self._adj[node] = []
            self.node_metadata[node] = {}
        if self.node_metadata[node] == {}:
            self.node_metadata[node] = metadata

    def add_nodes(self, node_list: list, metadata=None):
        for node in node_list:
            try:
                self.add_node(node, metadata[node] if metadata is not None else None)
            except KeyError:
                raise ValueError(
                    "The metadata dictionary must contain an entry for each node in the node list."
                )

    def is_weighted(self):
        """
        Check if the hypergraph is weighted.

        Returns
        -------
        bool
            True if the hypergraph is weighted, False otherwise.
        """
        return self._weighted

    def get_weight(self, edge, time):
        if (time, edge) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[(time, edge)]
        return self._weights[e_id]

    def set_weight(self, edge, time, weight):
        if not self._weighted and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )
        if (time, edge) not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        e_id = self._edge_list[(time, edge)]
        self._weights[e_id] = weight

    def get_nodes(self, metadata=False):
        if metadata:
            return self.node_metadata
        return list(self.node_metadata.keys())

    def add_edge(self, edge, time, weight=None, metadata=None):
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

        _edge = tuple(sorted(edge))
        edge = (t, _edge)

        if edge not in self._edge_list:
            e_id = self.next_edge_id
            self.reverse_edge_list[e_id] = edge
            self._edge_list[edge] = e_id
            self.next_edge_id += 1
            self._weights[e_id] = weight
        elif edge in self._edge_list and self._weighted:
            self._weights[self._edge_list[edge]] += weight

        e_id = self._edge_list[edge]

        if metadata is None:
            metadata = {}
        self.edge_metadata[e_id] = metadata

        for node in _edge:
            self.add_node(node)

        for node in _edge:
            self._adj[node].append(e_id)

    def add_edges(self, edge_list, time_list, weights=None, metadata=None):
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

    def get_edges(self, time_window=None, metadata=False):
        edges = []
        if time_window is None:
            if metadata:
                return {
                    edge: self.edge_metadata[edge] for edge in self._edge_list.keys()
                }
            else:
                return list(self._edge_list.keys())
        elif isinstance(time_window, tuple) and len(time_window) == 2:
            for _t, _edge in list(sorted(self._edge_list.keys())):
                if time_window[0] <= _t < time_window[1]:
                    edges.append((_t, _edge))
            if metadata:
                return {
                    edge: self.get_edge_metadata(edge[0], edge[1]) for edge in edges
                }
            else:
                return edges
        else:
            raise ValueError("Time window must be a tuple of length 2")

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
                Hypergraph_t.add_node(node, metadata=self.node_metadata[node])

            # Store the finalized hypergraph for this window
            aggregated[num_windows_created] = Hypergraph_t
            num_windows_created += 1

            # Advance to the next time window
            t_start = t_end
            t_end += time_window
            edges_in_window = []  # Reset for the next window

        return aggregated

    def add_attr_to_node_metadata(self, node, field, value):
        if node not in self.node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self.node_metadata[node][field] = value

    def add_attr_to_edge_metadata(self, edge, time, field, value):
        edge = tuple(sorted(edge))
        if edge not in self.edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self.edge_metadata[self._edge_list[(time, edge)]][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self.node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        del self.node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge, time, field):
        edge = tuple(sorted(edge))
        if edge not in self.edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        del self.edge_metadata[self._edge_list[(time, edge)]][field]
