import hoinetx as hnx
from hoinetx.linalg.linalg import binary_incidence_matrix, incidence_matrix  # fix ugly import

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

