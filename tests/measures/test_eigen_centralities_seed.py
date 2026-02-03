from hypergraphx import Hypergraph
from hypergraphx.measures.eigen_centralities import CEC_centrality


def test_cec_centrality_reproducible_with_seed():
    hg = Hypergraph(edge_list=[(0, 1, 2), (1, 2, 3)], weighted=False)
    c1 = CEC_centrality(hg, tol=1e-6, max_iter=50, seed=0)
    c2 = CEC_centrality(hg, tol=1e-6, max_iter=50, seed=0)
    assert c1 == c2
