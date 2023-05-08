from hypergraphx import Hypergraph
from hypergraphx.core.meta_handler import MetaHandler


class MultiplexHypergraph:

    def __init__(self):
        self.layers = {}
        self.__attr = MetaHandler()

    def add_layer(self, layer_name, hypergraph: Hypergraph, attr=None):
        self.layers[layer_name] = hypergraph
        self.__attr.set_attr(layer_name, attr)

    def get_layer(self, layer_name):
        return self.layers[layer_name]

    def del_layer(self, layer_name):
        del self.layers[layer_name]

    def get_layers(self):
        return list(self.layers.keys())

    def get_layer_attr(self, layer_name):
        return self.__attr.get_attr(layer_name)

    def set_layer_attr(self, layer_name, attr):
        self.__attr.set_attr(layer_name, attr)

    def get_nodes(self):
        nodes = set()
        for layer in self.layers.values():
            nodes.update(layer.get_nodes())
        return list(nodes)

    def num_layers(self):
        return len(self.layers)

    def clear(self):
        self.layers.clear()

    def degree(self, node, order=None, return_sum=False):
        degree = {}
        for layer_name in self.layers:
            layer = self.layers[layer_name]
            degree[layer_name] = layer.degree(node, order=order)
        if return_sum:
            return sum(degree.values())
        else:
            return degree

    def degree_sequence(self, order=None, return_sum=False):
        degree = {}
        for node in self.get_nodes():
            degree[node] = self.degree(node, order=order, return_sum=return_sum)
        return degree

    def aggregated_hypergraph(self):
        h = Hypergraph()
        for layer in self.layers.values():
            h.add_edges(layer._edge_list.keys())
        return h

    def edge_overlap(self, edge):
        edge = tuple(sorted(edge))
        overlap = 0
        for layer in self.layers.values():
            if edge in layer._edge_list:
                overlap += layer.get_weight(edge)
        return overlap

    def __str__(self):
        title = "Multiplex hypergraph with {} layers.\n".format(self.num_layers())
        details = "Layers: {}".format(self.get_layers())
        for layer in self.layers.values():
            details += "\n" + str(layer)
        return title + details

    def __repr__(self):
        pass

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, layer_name):
        return self.layers[layer_name]

    def __delitem__(self, layer_name):
        del self.layers[layer_name]

    def __setitem__(self, layer_name, hypergraph):
        self.layers[layer_name] = hypergraph

    def __contains__(self, layer_name):
        return layer_name in self.layers
