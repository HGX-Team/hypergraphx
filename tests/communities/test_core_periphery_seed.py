from hypergraphx import Hypergraph
from hypergraphx.communities.core_periphery.model import core_periphery


def test_core_periphery_reproducible_with_seed():
    hg = Hypergraph(edge_list=[(0, 1, 2), (1, 2, 3), (3, 4)], weighted=False)
    out1 = core_periphery(hg, N_ITER=2, seed=0)
    out2 = core_periphery(hg, N_ITER=2, seed=0)
    assert out1 == out2
