import hoinetx as hnx
from hoinetx.generation.scale_free import scale_free
from hoinetx.linalg import *
from hoinetx.representations.projections import bipartite, clique_projection
from hoinetx.generation.random import *
from hoinetx.readwrite.save import save_hypergraph
from hoinetx.readwrite.load import load_hypergraph

#H = hnx.Hypergraph([(1, 3), (1, 4), (1, 2), (5, 6, 7, 8), (1, 2, 3)])
#save(H, "test.hnx", "pickle")

H = load_hypergraph("test.hnx", "pickle")
print(H)