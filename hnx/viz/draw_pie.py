import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from hnx import Hypergraph
from scipy import sparse
import sknetwork as skn
from sknetwork.data import from_edge_list
from sknetwork.visualization import svg_graph
from sknetwork.clustering import Louvain
from sknetwork.utils import Bunch
from sknetwork.embedding import Spring


def _draw_from_matrix(matrix, number_of_items_to_pick=100):
    from numpy.random import choice
    candidates_names = [(i, j) for i in range(matrix.shape[0]-1) for j in range(i+1, matrix.shape[1])]
    candidates = [i for i in range(len(candidates_names))]
    num_candidates_non_zero = len([matrix[i][j] for i, j in candidates_names if matrix[i][j] != 0])
    number_of_items_to_pick = min(number_of_items_to_pick, num_candidates_non_zero)
    p = [matrix[i][j] for i, j in candidates_names]
    S = sum(p)
    p = [i / S for i in p]
    draw = choice(candidates, number_of_items_to_pick, p=p, replace=False)
    edges = [candidates_names[i] for i in draw]
    return edges


def get_proportion_hyperedges(hg: Hypergraph, max_size=5):
    """
    Returns the proportion of hyperedges that contain the node
    """
    p = []
    for node in hg.get_nodes():
        tmp = []
        for size in range(2, max_size + 2):
            if size == max_size + 1:
                tmp.append((len([edge for edge in hg.get_incident_edges(node)]) - sum(tmp)) / len(hg.get_incident_edges(node)))
            else:
                tmp.append(len([edge for edge in hg.get_incident_edges(node, size=size)]) / len(hg.get_incident_edges(node)))
        p.append(tmp)
    return p


def draw_pie(hg: Hypergraph):
    node2id = {node: i for i, node in enumerate(hg.get_nodes())}
    id2node = {i: node for i, node in enumerate(hg.get_nodes())}

    # create a matrix of size (n_nodes, n_nodes) initialized with zeros
    matrix = np.zeros((len(hg.get_nodes()), len(hg.get_nodes())))

    for node in hg.get_nodes():
        for edge in hg.get_incident_edges(node):
            for node2 in edge:
                if node != node2:
                    matrix[node2id[node], node2id[node2]] += 1
                    matrix[node2id[node2], node2id[node]] += 1

    edges = _draw_from_matrix(matrix, number_of_items_to_pick=10)
    matrix = np.zeros((len(hg.get_nodes()), len(hg.get_nodes())))
    for edge in edges:
        matrix[edge[0], edge[1]] = 1
        matrix[edge[1], edge[0]] = 1

    matrix = matrix.tolist()
    adjacency = sparse.csr_matrix(matrix)

    # probabilities
    max_size = 2
    membership = sparse.csr_matrix(get_proportion_hyperedges(hg, max_size=max_size))
    COLORS = ['red', 'yellow', 'green', 'blue']
    label_colors = [COLORS[i] for i in range(max_size-1)]
    label_colors.append('black')
    print(membership.todense())

    a = svg_graph(adjacency,
                  membership=membership,
                  node_size=10,
                  filename='pie',
                  label_colors=label_colors)


