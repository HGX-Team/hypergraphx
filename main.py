import hoinetx as hnx
from hoinetx.generation.scale_free import scale_free
from hoinetx.linalg import *
from hoinetx.representations.projections import bipartite, clique_projection
from hoinetx.generation.random import *
from hoinetx.readwrite.save import save_hypergraph
from hoinetx.readwrite.load import load_hypergraph
from hoinetx.viz.draw_projections import draw_bipartite, draw_clique

H = hnx.Hypergraph([(1, 2, 3), (1, 2), (4, 3)], weights=[1, 2, 3])
draw_clique(H)