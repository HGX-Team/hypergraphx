import hoinetx as hnx
from hoinetx.generation.scale_free import scale_free
from hoinetx.linalg import *
from hoinetx.representations.projections import bipartite, clique_projection
from hoinetx.generation.random import *
from hoinetx.readwrite.save import save_hypergraph
from hoinetx.readwrite.load import load_hypergraph

H = hnx.Hypergraph([(1, 2, 3), (1,2), (4,3)], weights=[1, 2, 3, 4, 5])
print(H.is_connected(order=1))
