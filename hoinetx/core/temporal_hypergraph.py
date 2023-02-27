import bisect
from hoinetx.core import Hypergraph
from hoinetx.core.attribute_handler import AttributeHandler


class TemporalHypergraph:
    def __init__(self):
        self.__attr = AttributeHandler()
        self.edges = []

    def add_edge(self, edge, attr=None):
        bisect.insort(self, element)

    def add_edges(self, edges, attr=None):
        pass

    def add_node(self, node, attr=None):
        pass

    def add_nodes(self, nodes, attr=None):
        pass

    def del_edge(self, edge):
        pass

    def del_edges(self, edges):
        pass

    def get_edges(self, time_window=None):
        pass

    def aggregate(self, time_window=None):
        pass






