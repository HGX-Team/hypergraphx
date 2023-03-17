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

#H = hnx.Hypergraph([(1, 2, 3), (1, 3, 4, 5), (1, 2), (3, 4)])
#draw_pie(H)
H = load_hypergraph("test_data/workspace/workspace.json", "json")
print(H.get_meta(875))