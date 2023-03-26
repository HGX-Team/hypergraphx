import hnx as hnx
from hnx.generation.scale_free import scale_free_hypergraph
from hnx.linalg import *
from hnx.representations.projections import bipartite_projection, clique_projection
from hnx.generation.random import *
from hnx.readwrite.save import save_hypergraph
from hnx.readwrite.load import load_hypergraph
from hnx.viz.draw_projections import draw_bipartite, draw_clique
from hnx.viz.draw_pie import draw_pie
import sklearn
from hnx.measures.degree import *

H = load_hypergraph("test_data/hs/hs.pickle", "pickle")
print(degree_correlation(H))