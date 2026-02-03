import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.dynamics.contagion import simplicial_contagion
from hypergraphx.dynamics.randwalk import random_walk


def test_simplicial_contagion_reproducible_with_seed():
    h = Hypergraph(edge_list=[(0, 1), (0, 1, 2)], weighted=False)
    I0 = {0: 1, 1: 0, 2: 0}

    out1 = simplicial_contagion(h, I0, T=20, beta=0.5, beta_D=0.5, mu=0.1, seed=0)
    out2 = simplicial_contagion(h, I0, T=20, beta=0.5, beta_D=0.5, mu=0.1, seed=0)
    assert np.allclose(out1, out2)


def test_random_walk_reproducible_with_seed():
    h = Hypergraph(edge_list=[(0, 1), (1, 2), (0, 1, 2)], weighted=False)

    p1 = random_walk(h, 0, 25, seed=0)
    p2 = random_walk(h, 0, 25, seed=0)
    assert p1 == p2
