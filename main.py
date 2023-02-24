import hoinetx as hnx
from hoinetx.linalg.linalg import *  # fix ugly import
from hoinetx.representations.projections import bipartite, clique_projection
from hoinetx.generation.random import random_hypergraph

H = hnx.Hypergraph([(1, 2, 3), (1, 4, 5), (1, 2), (5, 6, 7, 8)])
print(H)
print("Weights: {}".format(H.get_weights()))
binary_incidence = binary_incidence_matrix(H, shape=(H.num_nodes(), H.num_edges()))
print(binary_incidence)

H = hnx.Hypergraph([(1, 2, 3), (1, 4, 5), (1, 2), (5, 6, 7, 8), (1, 2, 3)])
print(H)
print("Weights: {}".format(H.get_weights()))
incidence = incidence_matrix(H, shape=(H.num_nodes(), H.num_edges()))
print(incidence)

print(H.degree_sequence())
print(H.degree_sequence(order=3))

H = hnx.Hypergraph([(1, 2, 3, 4), (1, 4)])
print(clique_projection(H).edges)

H = hnx.Hypergraph([(1, 2, 3), (1, 4)])
print(bipartite(H)[0].edges)

H = hnx.Hypergraph([(1, 2, 3), (1, 4, 5), (1, 2), (5, 6, 7, 8)], weights=[1, 2, 3, 4], weighted=True)
print(H)
print("Weights: {}".format(H.get_weights()))

for edge in H:
    print(edge)

H = random_hypergraph(10, {2: 5, 3: 2, 4: 1})
print(H)

print(incidence_matrices_all_orders(H))
