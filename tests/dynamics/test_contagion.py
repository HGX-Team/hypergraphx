import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.dynamics.contagion import simplicial_contagion


def test_simplicial_contagion_no_spread():
    """Test contagion with zero infection stays constant."""
    np.random.seed(0)
    hg = Hypergraph(edge_list=[(0, 1), (1, 2, 3)])
    I_0 = {0: 1, 1: 0, 2: 0, 3: 0}

    infected = simplicial_contagion(hg, I_0, T=4, beta=0.0, beta_D=0.0, mu=0.0)

    assert infected[0] == 0.25
    assert infected[-1] == 0.25


def test_simplicial_contagion_recovery():
    """Test contagion recovery can reduce infections."""
    np.random.seed(1)
    hg = Hypergraph(edge_list=[(0, 1), (1, 2, 3)])
    I_0 = {0: 1, 1: 1, 2: 0, 3: 0}

    infected = simplicial_contagion(hg, I_0, T=5, beta=0.0, beta_D=0.0, mu=1.0)

    assert infected[0] == 0.5
    assert infected[1] == 0.0
