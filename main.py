import hoinetx as hnx
from hoinetx.generation.scale_free import scale_free
from hoinetx.linalg.linalg import *  # fix ugly import
from hoinetx.representations.projections import bipartite, clique_projection
from hoinetx.generation.random import *

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
print("OK")
H = random_hypergraph(10, {2: 5, 3: 2, 4: 1})
print(H)

H = scale_free(10, {2: 10, 3: 6, 4: 3}, {2: 2, 3: 1, 4: 1}, correlated=True, corr_value=0.1)
print(H.degree_sequence(size=2))
print(H.degree_sequence(size=3))
print(H.degree_sequence(size=4))
