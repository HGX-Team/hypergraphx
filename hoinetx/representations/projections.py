import networkx as nx
from hoinetx.core.hypergraph import Hypergraph


def bipartite(h: Hypergraph):
    g = nx.Graph()
    id_to_obj = {}
    obj_to_id = {}
    idx = 0

    for node in h.get_nodes():
        id_to_obj[idx] = node
        obj_to_id[node] = idx
        idx += 1

    for edge in h.edge_list:
        edge = tuple(sorted(edge))
        obj_to_id[edge] = idx
        id_to_obj[idx] = edge
        idx += 1
        g.add_node(obj_to_id[edge])

        for node in edge:
            g.add_edge(obj_to_id[edge], obj_to_id[node])

    return g, id_to_obj


def clique_projection(h: Hypergraph):
    g = nx.Graph()

    for edge in h.edge_list:
        for i in range(len(edge)-1):
            for j in range(i+1, len(edge)):
                g.add_edge(edge[i], edge[j])

    return g


def line_graph(h: Hypergraph, distance='intersection', s=1, weighted=False):
    def intersection(a, b):
        return len(a.intersection(b))

    def jaccard(a, b):
        return len(a.intersection(b)) / len(a.union(b))

    def distance(a, b):
        if distance == 'intersection':
            return intersection(a, b)
        if distance == 'jaccard':
            return jaccard(a, b)

    edges = h.edge_list
    adj = h.get_adj_nodes()

    edge_to_id = {}
    id_to_edge = {}
    cont = 0
    for e in edges:
        e = tuple(sorted(e))
        edge_to_id[e] = cont
        id_to_edge[cont] = e
        cont += 1

    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(h))])

    vis = {}

    for n in adj:
        for i in range(len(adj[n]) - 1):
            for j in range(i + 1, len(adj[n])):
                k = tuple(sorted((edge_to_id[adj[n][i]], edge_to_id[adj[n][j]])))
                e_i = set(adj[n][i])
                e_j = set(adj[n][j])
                if k not in vis:
                    w = distance(e_i, e_j)
                    if w >= s:
                        if weighted:
                            g.add_edge(edge_to_id[adj[n][i]], edge_to_id[adj[n][j]], weight=w)
                        else:
                            g.add_edge(edge_to_id[adj[n][i]], edge_to_id[adj[n][j]], weight=1)
                    vis[k] = True
    return g, id_to_edge
