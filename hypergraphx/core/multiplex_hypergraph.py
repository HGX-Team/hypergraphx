from hypergraphx import Hypergraph


class MultiplexHypergraph:

    def __init__(self):
        self.layers = {}

    def get_hypergraph_metadata(self):
        metadata = {}
        for layer_name in self.layers:
            metadata[layer_name] = self.layers[layer_name].get_hypergraph_metadata()
        return metadata

    def add_layer(self, layer_name, hypergraph: Hypergraph):
        self.layers[layer_name] = hypergraph

    def get_layer(self, layer_name):
        return self.layers[layer_name]

    def del_layer(self, layer_name):
        del self.layers[layer_name]

    def get_layers(self):
        return list(self.layers.keys())

    def get_nodes(self):
        nodes = {}
        in_layers = {}
        for layer in self.layers.values():
            layer_nodes = layer.get_nodes(metadata=True)
            for node in layer_nodes:
                if node not in nodes:
                    nodes[node] = layer_nodes[node]
                if node not in in_layers:
                    in_layers[node] = []
                in_layers[node].append(layer)
        for node in nodes:
            nodes[node]['in_layers'] = in_layers[node]
        return list(nodes)
    
    def get_edges(self):
        edges = {}
        for layer in self.layers.values():
            layer_edges = layer.get_edges(metadata=True)
            for edge in layer_edges:
                if edge not in edges:
                    edges[edge] = layer_edges[edge]
                    edges[edge]['layer'] = layer
        return list(edges)

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
