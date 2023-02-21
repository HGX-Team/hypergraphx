from hoinetx.core.attribute_handler import AttributeHandler


class Hypergraph:

    def __init__(self, edge_list=[], weighted=False, weights=None):
        self.__attr = AttributeHandler()
        self.edge_list = {}
        self.weighted = weighted
        self.node_list = set()
        self.edges_by_order = {}
        self.__max_order = 0

        if weighted and weights is not None:
            if len(set(edge_list)) != len(edge_list):
                raise ValueError("If weights are provided, the edge list must not contain repeated edges.")
            if len(edge_list) != len(weights):
                raise ValueError("The number of edges and weights must be the same.")

        self.add_edges(edge_list, weights=weights)

    def is_weighted(self):
        return self.weighted

    def add_edge(self, edge, w=None):
        edge = tuple(sorted(edge))
        idx = self.__attr.get_id(edge)
        order = len(edge)

        if order > self.__max_order:
            self.__max_order = order

        if order not in self.edges_by_order:
            self.edges_by_order[order] = [idx]
        else:
            self.edges_by_order[order].append(idx)

        if w is None:
            if edge in self.edge_list and self.weighted:
                self.edge_list[edge] += 1
            else:
                self.edge_list[edge] = 1
        else:
            self.edge_list[edge] = w

        for node in edge:
            self.node_list.add(node)

    def add_edges(self, edge_list, weights=None):
        i = 0
        for edge in edge_list:
            self.add_edge(edge, w=weights[i] if self.weighted and weights is not None else None)
            i += 1

    def del_edge(self, edge, force=False):
        try:
            edge = tuple(sorted(edge))
            self.edge_list[edge] -= 1
            if self.edge_list[edge] == 0 or force:
                del self.edge_list[edge]
                order = len(edge)
                idx = self.__attr.get_id(edge)
                self.edges_by_order[order].remove(idx)
                self.node_list = set()
                self.__max_order = 0
                for edge in self.edge_list:
                    order = len(edge)
                    if order > self.__max_order:
                        self.__max_order = order
                    for node in edge:
                        self.node_list.add(node)
        except KeyError:
            print("Edge {} not in hypergraph.".format(edge))

    def max_order(self):
        return self.__max_order

    def get_nodes(self):
        return list(self.node_list)

    def get_edges(self):
        return list(self.edge_list.keys())

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

    def get_edges_by_order(self, order):
        try:
            return Hypergraph([self.__attr.get_obj(idx) for idx in self.edges_by_order[order]])
        except KeyError:
            return Hypergraph()

    def degree(self, node, order=None):
        if order is None:
            return sum([1 for edge in self.edge_list if node in edge])
        else:
            return sum([1 for edge in self.edge_list if node in edge and len(edge) == order])

    def degree_sequence(self, order=None):
        if order is None:
            return {node: self.degree(node) for node in self.node_list}
        else:
            return {node: self.degree(node, order) for node in self.node_list}

    def is_connected(self):
        pass

    def get_adj_node(self, node):
        return [edge for edge in self.edge_list if node in edge]

    def get_adj_nodes(self):
        return {node: self.get_adj_node(node) for node in self.node_list}

    def clear(self):
        self.edge_list.clear()
        self.node_list.clear()
        self.edges_by_order.clear()
        self.__max_order = 0
        self.__attr.clear()

    def __str__(self):
        title = "Hypergraph with {} nodes and {} edges.\n".format(self.num_nodes(), self.num_edges())
        details = "Edge list: {}".format(list(self.edge_list.keys()))
        return title + details

    def __len__(self):
        return len(self.edge_list)

    def __iter__(self):
        return iter(self.edge_list)


