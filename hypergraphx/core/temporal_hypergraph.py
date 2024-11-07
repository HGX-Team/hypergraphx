from hypergraphx import Hypergraph


class TemporalHypergraph:
    def __init__(self, edges):
        self.edges = {}
        self._nodes = set()
        self.add_edges(edges)

    def add_edge(self, edge):
        if not isinstance(edge, tuple):
            raise TypeError('Edge must be a tuple')
        if len(edge) != 2:
            raise ValueError('Edge must be a tuple of length 2')
        t = edge[0]
        _edge = tuple(sorted(edge[1]))
        if not isinstance(_edge, tuple):
            raise TypeError('Edge must be a tuple')
        if len(_edge) < 2:
            raise ValueError('Edge must be a tuple of length 2 or more')
        if t < 0:
            raise ValueError('Time must be a positive integer')
        for node in _edge:
            self._nodes.add(node)
        if t not in self.edges:
            self.edges[t] = set([_edge])
        else:
            self.edges[t].add(_edge)

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node):
        self._nodes.add(node)

    def add_nodes(self, nodes):
        for node in nodes:
            self._nodes.add(node)

    def remove_edge(self, edge):
        t = edge[0]
        _edge = tuple(sorted(edge[1]))
        if t in self.edges:
            if _edge in self.edges[t]:
                self.edges[t].remove(_edge)

    def remove_edges(self, edges):
        for edge in edges:
            self.remove_edge(edge)

    def get_edges(self, time_window=None):
        edges = []
        if time_window is None:
            for t in sorted(self.edges):
                for edge in self.edges[t]:
                    edges.append((t, edge))
        elif isinstance(time_window, tuple) and len(time_window) == 2:
            for t in sorted(self.edges):
                if time_window[0] <= t < time_window[1]:
                    for edge in self.edges[t]:
                        edges.append((t, edge))
        else:
            for edge in self.edges[time_window]:
                edges.append((time_window, edge))
        return edges

    def aggregate(self, time_window=None):
        aggregated = {}
        if not isinstance(time_window, int):
            raise TypeError('Time window must be an integer')
        t = 0
        node_list = list(self.get_nodes())
        while t < max(self.edges):
            aggregated[t] = set()
            for edge in self.get_edges((t, t + time_window)):
                aggregated[t].add(edge[1])
            # I changed this line of code so that the hypergraph at each time stamp has the same size
            Hypergraph_t = Hypergraph(aggregated[t])
            Hypergraph_t.add_nodes(node_list)
            aggregated[t] = Hypergraph_t
            t += time_window
        return aggregated

    def get_nodes(self):
        return self._nodes

    def __str__(self):
        for edge in self.edges:
            print(edge)
        return ''






