from hoinetx.core.attribute_handler import AttributeHandler


class Hypergraph:

    def __init__(self, edge_list=[]):
        self.__attr = AttributeHandler()
        self.edge_list = {}
        self.node_list = set()

        for edge in edge_list:
            edge = tuple(sorted(edge))
            _ = self.__attr.get_id(edge)

            if edge not in self.edge_list:
                self.edge_list[edge] = 1
            else:
                self.edge_list[edge] += 1

            for node in edge:
                self.node_list.add(node)

    def max_order(self):
        return max([len(edge) for edge in self.edge_list.keys()])

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
        return Hypergraph([edge for edge in self.edge_list.keys() if len(edge) == order])

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
