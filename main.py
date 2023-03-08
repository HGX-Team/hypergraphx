import hoinetx as hnx
from hoinetx.generation.scale_free import scale_free
from hoinetx.linalg import *
from hoinetx.representations.projections import bipartite, clique_projection
from hoinetx.generation.random import *

H = hnx.Hypergraph([(1, 3), (1, 4), (1, 2), (5, 6, 7, 8), (1, 2, 3)])
print(laplacian_matrix_by_order(H, 1))
