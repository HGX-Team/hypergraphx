from hoinetx.core.attribute_handler import AttributeHandler


class Hypergraph:

    def __init__(self, edge_list=[]):
        self.__attr = AttributeHandler()
        self.edge_list = {}
        self.node_list = set()
        self.edges_by_order = {}
        self.__max_order = 0

        for edge in edge_list:
            edge = tuple(sorted(edge))
            idx = self.__attr.get_id(edge)
            order = len(edge)

            if order > self.__max_order:
                self.__max_order = order

            if order not in self.edges_by_order:
                self.edges_by_order[order] = [idx]
            else:
                self.edges_by_order[order].append(idx)

            if edge not in self.edge_list:
                self.edge_list[edge] = 1
            else:
                self.edge_list[edge] += 1

            for node in edge:
                self.node_list.add(node)

    def max_order(self):
        return self.__max_order

    def num_nodes(self):
        return len(self.node_list)

    def num_edges(self):
        return len(self.edge_list)

    def get_weight(self, edge):
        return self.edge_list[tuple(sorted(edge))]

    def get_weights(self):
        return list(self.edge_list.values())

    def get_sizes(self):
        return [len(edge) for edge in self.edge_list.keys()]

    def get_attr(self, obj):
        return self.__attr.get_attr(obj)

    def set_attr(self, obj, attr):
        self.__attr.set_attr(obj, attr)

    def check_edge(self, edge):
        return tuple(sorted(edge)) in self.edge_list

    def filter_by_order(self, order):
        try:
            return Hypergraph([self.__attr.get_obj(idx) for idx in self.edges_by_order[order]])
        except KeyError:
            return Hypergraph()

    def __str__(self):
        title = "Hypergraph with {} nodes and {} edges.\n".format(self.num_nodes(), self.num_edges())
        details = "Edge list: {}".format(list(self.edge_list.keys()))
        return title + details

    def __repr__(self):
        pass

    def __len__(self):
        return len(self.edge_list)

    def __iter__(self):
        return iter(self.edge_list)
