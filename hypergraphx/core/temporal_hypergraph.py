from hypergraphx import Hypergraph


class TemporalHypergraph:
    def __init__(self, edge_list=None, hypergraph_metadata=None, weighted=False, weights=None, edge_metadata=None):
        if hypergraph_metadata is None:
            self.hypergraph_metadata = {}
        else:
            self.hypergraph_metadata = hypergraph_metadata
        self.hypergraph_metadata['weighted'] = weighted
        self._weighted = weighted
        self._edge_list = {}
        self.node_metadata = {}
        self.edge_metadata = {}

        if edge_list is not None:
            self.add_edges(edge_list)

    def add_node(self, node, metadata=None):
        if node not in self.node_metadata:
            if metadata is not None:
                self.node_metadata[node] = metadata
            else:
                self.node_metadata[node] = {}

    def add_nodes(self, node_list: list, metadata=None):
        for node in node_list:
            try:
                self.add_node(node, metadata[node] if metadata is not None else None)
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

    def get_nodes(self, metadata=False):
        if metadata:
            return self.node_metadata
        return list(self.node_metadata.keys())

    def add_edge(self, edge, weight=None, metadata=None):
        if not isinstance(edge, tuple):
            raise TypeError('Edge must be a tuple')
        if len(edge) != 2:
            raise ValueError('Edge must be a tuple of length 2')
        if not isinstance(edge[0], int):
            raise TypeError('Time must be an integer')
        if self._weighted and weight is None:
            raise ValueError(
                "If the hypergraph is weighted, a weight must be provided."
            )
        if not self._weighted and weight is not None:
            raise ValueError(
                "If the hypergraph is not weighted, no weight must be provided."
            )

        t = edge[0]

        if t < 0:
            raise ValueError('Time must be a positive integer')

        _edge = tuple(sorted(edge[1]))

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

    def add_edges(self, edge_list, weights=None, metadata=None):
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

    def get_edges(self, time_window=None, metadata=False):
        edges = []
        if time_window is None:
           if metadata:
                return self.edge_metadata
           else:
                return list(self._edge_list.keys())
        elif isinstance(time_window, tuple) and len(time_window) == 2:
            for _t, _edge in list(sorted(self._edge_list.keys())):
                if time_window[0] <= _t < time_window[1]:
                    edges.append((_t, _edge))
            if metadata:
                return {edge: self.edge_metadata[edge] for edge in edges}
            else:
                return edges
        else:
            raise ValueError('Time window must be a tuple of length 2')

    def aggregate(self, time_window):
        if not isinstance(time_window, int) or time_window <= 0:
            raise TypeError('Time window must be a positive integer')

        aggregated = {}
        node_list = self.get_nodes()

        # Sort edges by time
        sorted_edges = sorted(self.get_edges())

        # Initialize time window boundaries
        t_start = 0
        t_end = time_window
        edges_in_window = []
        num_windows_created = 0

        for edge in sorted_edges:
            time, edge_nodes = edge

            # If the current edge falls outside the current window, finalize this window
            if time >= t_end:
                # Finalize the current window
                Hypergraph_t = Hypergraph()

                # Add edges in this window to the hypergraph
                for _, e in edges_in_window:
                    Hypergraph_t.add_edge(e)

                # Add all nodes to ensure node consistency
                for node in node_list:
                    Hypergraph_t.add_node(node)

                # Store the finalized hypergraph and move to the next window
                aggregated[num_windows_created] = Hypergraph_t
                num_windows_created += 1

                # Advance to the next time window
                t_start = t_end
                t_end += time_window
                edges_in_window = []

            # Add the current edge to the active window
            edges_in_window.append((time, edge_nodes))

        # Finalize the last window if any edges remain
        if edges_in_window:
            Hypergraph_t = Hypergraph()
            for _, e in edges_in_window:
                Hypergraph_t.add_edge(e)
            for node in node_list:
                Hypergraph_t.add_node(node)
            aggregated[num_windows_created] = Hypergraph_t
            num_windows_created += 1
        return aggregated


