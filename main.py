import hnx as hnx
from hnx.generation.scale_free import scale_free
from hnx.linalg import *
from hnx.representations.projections import bipartite, clique_projection
from hnx.generation.random import *
from hnx.readwrite.save import save_hypergraph
from hnx.readwrite.load import load_hypergraph
from hnx.viz.draw_projections import draw_bipartite, draw_clique

H = hnx.Hypergraph([(1, 2, 3), (1, 2), (4, 3)], weights=[1, 2, 3])
draw_clique(H)