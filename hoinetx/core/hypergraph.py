import copy
from typing import Optional, Tuple

from scipy import sparse

from hoinetx.core.attribute_handler import AttributeHandler


class Hypergraph:

    def __init__(self, edge_list=None, weighted=False, weights=None):
        self._attr = AttributeHandler()
        self.edge_list = {}
        self._weighted = weighted
        self._edges_by_order = {}
        self._neighbors = {}
        self._adj = {}
        self._max_order = 0
        self.add_edges(edge_list, weights=weights)

    def get_neighbors(self, node, order=None, size=None):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return self._neighbors[node]
        else:
            if order is None:
                order = size - 1
            neigh = set()
            edges = self.get_adj_edges(node, order=order)
            for edge in edges:
                neigh.update(edge)
            neigh.remove(node)
            return neigh

    def get_adj_edges(self, node, order=None, size=None):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return [self._attr.get_obj(idx) for idx in self._adj[node]]
        else:
            if order is None:
                order = size - 1
            edges = []
            for idx in self._adj[node]:
                edge = self._attr.get_obj(idx)
                if len(edge) == order + 1:
                    edges.append(edge)
            return edges

    def add_node(self, node):
        if node not in self._neighbors:
            self._neighbors[node] = set()
            self._adj[node] = set()

    def add_nodes(self, node_list):
        for node in node_list:
            self.add_node(node)

    def is_weighted(self):
        return self._weighted

    def add_edge(self, edge, weight=None):
        if self._weighted and weight is None:
            raise ValueError("If the hypergraph is weighted, a weight must be provided.")
        if not self._weighted and weight is not None:
            raise ValueError("If the hypergraph is not weighted, no weight must be provided.")

        edge = tuple(sorted(edge))
        idx = self._attr.get_id(edge)
        order = len(edge) - 1

        if order > self._max_order:
            self._max_order = order

        if order not in self._edges_by_order:
            self._edges_by_order[order] = [idx]
        else:
            self._edges_by_order[order].append(idx)

        if weight is None:
            if edge in self.edge_list and self._weighted:
                self.edge_list[edge] += 1
            else:
                self.edge_list[edge] = 1
        else:
            self.edge_list[edge] = weight

        for node in edge:
            self.add_node(node)
            self._adj[node].add(idx)

        for i in range(len(edge)):
            for j in range(i + 1, len(edge)):
                self._neighbors[edge[i]].add(edge[j])
                self._neighbors[edge[j]].add(edge[i])

    def add_edges(self, edge_list, weights=None):
        if self._weighted and weights is not None:
            if len(set(edge_list)) != len(edge_list):
                raise ValueError("If weights are provided, the edge list must not contain repeated edges.")
            if len(edge_list) != len(weights):
                raise ValueError("The number of edges and weights must be the same.")

        if weights is not None and not self._weighted:
            raise ValueError("If weights are provided, the hypergraph must be weighted.")

        i = 0
        if edge_list is not None:
            for edge in edge_list:
                self.add_edge(edge, weight=weights[i] if self._weighted and weights is not None else None)
                i += 1

    def _compute_neighbors(self, node):
        neighbors = set()
        for edge in self._adj[node]:
            neighbors = neighbors.union(self._attr.get_obj(edge))
        neighbors.remove(node)
        return neighbors

    def del_edge(self, edge, force=False):
        try:
            edge = tuple(sorted(edge))
            self.edge_list[edge] -= 1
            if self.edge_list[edge] == 0 or force:
                del self.edge_list[edge]
                order = len(edge) - 1
                idx = self._attr.get_id(edge)
                for node in edge:
                    self._adj[node].remove(idx)
                for node in edge:
                    self._neighbors[node] = self._compute_neighbors(node)
                self._edges_by_order[order].remove(idx)
                self._max_order = 0
                for edge in self.edge_list:
                    order = len(edge) - 1
                    if order > self._max_order:
                        self._max_order = order
        except KeyError:
            print("Edge {} not in hypergraph.".format(edge))

    def del_edges(self, edge_list, force=False):
        for edge in edge_list:
            self.del_edge(edge, force=force)

    def del_node(self, node, keep_edges=False):
        if not keep_edges:
            for edge in self._adj[node]:
                self.del_edge(self._attr.get_obj(edge))
        else:
            for edge in self._adj[node]:
                edge = self._attr.get_obj(edge)
                self.del_edge(edge)
                self.add_edge(tuple([n for n in edge if n != node]))
        del self._neighbors[node]
        del self._adj[node]

    def del_nodes(self, node_list, keep_edges=False):
        for node in node_list:
            self.del_node(node, keep_edges=keep_edges)

    def subhypergraph(self, nodes):
        return Hypergraph([edge for edge in self.edge_list if set(edge).issubset(set(nodes))])

    def max_order(self):
        return self._max_order

    def max_size(self):
        return self._max_order + 1

    def get_nodes(self):
        return list(self._neighbors.keys())

    def num_nodes(self):
        return len(self.get_nodes())

    def num_edges(self, order=None, size=None, up_to=False):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")

        if order is None and size is None:
            return len(self.edge_list)
        else:
            if size is not None:
                order = size - 1
            if not up_to:
                try:
                    return len(self._edges_by_order[order])
                except KeyError:
                    return 0
            else:
                s = 0
                for i in range(1, order + 1):
                    try:
                        s += len(self._edges_by_order[i])
                    except KeyError:
                        s += 0
                return s

    def get_weight(self, edge):
        try:
            return self.edge_list[tuple(sorted(edge))]
        except KeyError:
            raise ValueError("Edge {} not in hypergraph.".format(edge))

    def get_weights(self, order=None, size=None, up_to=False):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return list(self.edge_list.values())

        if size is not None:
            order = size - 1

        if not up_to:
            try:
                return [self.edge_list[self._attr.get_obj(idx)] for idx in self._edges_by_order[order]]
            except KeyError:
                return []
        else:
            w = []
            for i in range(1, order + 1):
                try:
                    w += [self.edge_list[self._attr.get_obj(idx)] for idx in self._edges_by_order[i]]
                except KeyError:
                    pass
            return w

    def get_sizes(self):
        return [len(edge) for edge in self.edge_list.keys()]

    def get_orders(self):
        return [len(edge) - 1 for edge in self.edge_list.keys()]

    def get_attr(self, obj):
        return self._attr.get_attr(obj)

    def set_attr(self, obj, attr):
        self._attr.set_attr(obj, attr)

    def check_edge(self, edge):
        return tuple(sorted(edge)) in self.edge_list

    def check_node(self, node):
        return node in self._neighbors

    def get_edges(self, ids=False, order=None, size=None, up_to=False, subhypergraph=False):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if ids and subhypergraph:
            raise ValueError("Cannot return subhypergraphs with ids.")

        if order is None and size is None:
            if not ids:
                edges = list(self.edge_list.keys())
            else:
                edges = [self._attr.get_id(edge) for edge in self.edge_list.keys()]
        else:
            if size is not None:
                order = size - 1
            if not up_to:
                if not ids:
                    edges = [self._attr.get_obj(edge) for edge in self._edges_by_order[order]]
                else:
                    edges = [edge for edge in self._edges_by_order[order]]
            else:
                edges = []
                if not ids:
                    for i in range(1, order + 1):
                        try:
                            edges += [self._attr.get_obj(edge) for edge in self._edges_by_order[i]]
                        except KeyError:
                            edges += []
                else:
                    for i in range(1, order + 1):
                        try:
                            edges += [edge for edge in self._edges_by_order[i]]
                        except KeyError:
                            edges += []

        if subhypergraph:
            return Hypergraph(edges)
        else:
            return edges

    def degree(self, node, order=None, size=None):
        from hoinetx.measures.degree import degree
        return degree(self, node, order=order, size=size)

    def degree_sequence(self, order=None, size=None):
        from hoinetx.measures.degree import degree_sequence
        return degree_sequence(self, order=order, size=size)

    def is_connected(self, size=None, order=None):
        from hoinetx.utils.cc import is_connected
        return is_connected(self, size=size, order=order)

    def connected_components(self, size=None, order=None):
        from hoinetx.utils.cc import connected_components
        return connected_components(self, size=size, order=order)

    def node_connected_component(self, node, size=None, order=None):
        from hoinetx.utils.cc import node_connected_component
        return node_connected_component(self, node, size=size, order=order)

    def num_connected_components(self, size=None, order=None):
        from hoinetx.utils.cc import num_connected_components
        return num_connected_components(self, size=size, order=order)

    def largest_component(self, size=None, order=None):
        from hoinetx.utils.cc import largest_component
        return largest_component(self, size=size, order=order)

    def largest_component_size(self, size=None, order=None):
        from hoinetx.utils.cc import largest_component_size
        return largest_component_size(self, size=size, order=order)

    def isolated_nodes(self, size=None, order=None):
        from hoinetx.utils.cc import isolated_nodes
        return isolated_nodes(self, size=size, order=order)

    def is_isolated(self, node, size=None, order=None):
        from hoinetx.utils.cc import is_isolated
        return is_isolated(self, node, size=size, order=order)

    def binary_incidence_matrix(self, shape: Optional[Tuple[int]] = None):
        from hoinetx.linalg import binary_incidence_matrix
        return binary_incidence_matrix(self, shape)

    def incidence_matrix(self, shape: Optional[Tuple[int]] = None):
        from hoinetx.linalg import incidence_matrix
        return incidence_matrix(self, shape)

    def adjacency_matrix(self) -> sparse.spmatrix:
        from hoinetx.linalg import adjacency_matrix
        return adjacency_matrix(self)

    def random_walk_adjacency(self) -> sparse.spmatrix:
        from hoinetx.linalg import random_walk_adjacency
        return random_walk_adjacency(self)

    def clear(self):
        self.edge_list.clear()
        self._neighbors.clear()
        self._edges_by_order.clear()
        self._max_order = 0
        self._attr.clear()

    def copy(self):
        return copy.deepcopy(self)

    def reindex_edges(self):
        pass

    def reindex_nodes(self):
        pass

    def __str__(self):
        title = "Hypergraph with {} nodes and {} edges.\n".format(self.num_nodes(), self.num_edges())
        details = "Edge list: {}".format(list(self.edge_list.keys()))
        return title + details

    def __len__(self):
        return len(self.edge_list)

    def __iter__(self):
        return iter(self.edge_list.items())


