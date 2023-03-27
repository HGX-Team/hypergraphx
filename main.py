import hypergraphx as hnx
from hypergraphx.generation.scale_free import scale_free_hypergraph
from hypergraphx.linalg import *
from hypergraphx.representations.projections import bipartite_projection, clique_projection
from hypergraphx.generation.random import *
from hypergraphx.readwrite.save import save_hypergraph
from hypergraphx.readwrite.load import load_hypergraph
from hypergraphx.viz.draw_projections import draw_bipartite, draw_clique
import sklearn
from hypergraphx.measures.degree import *

H = load_hypergraph("test_data/hs/hs.pickle", "pickle")
print(degree_correlation(H))