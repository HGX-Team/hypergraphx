import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.dynamics.utils import is_natural_coupling, is_all_to_all


def test_is_natural_coupling_true():
    """Test natural coupling detection with identical Jacobians."""
    JH = lambda x: np.eye(len(x))
    assert is_natural_coupling([JH, JH], dim=3, verbose=False) is True


def test_is_natural_coupling_false():
    """Test natural coupling detection with different Jacobians."""
    JH1 = lambda x: np.eye(len(x))
    JH2 = lambda x: np.zeros((len(x), len(x)))
    assert is_natural_coupling([JH1, JH2], dim=3, verbose=False) is False


def test_is_all_to_all_true():
    """Test all-to-all detection on complete size-2 hypergraph."""
    edges = [(0, 1), (0, 2), (1, 2)]
    hg = Hypergraph(edge_list=edges, weighted=False)

    assert is_all_to_all(hg, verbose=False) is True


def test_is_all_to_all_false():
    """Test all-to-all detection on incomplete hypergraph."""
    edges = [(0, 1), (0, 2)]
    hg = Hypergraph(edge_list=edges, weighted=False)

    assert is_all_to_all(hg, verbose=False) is False
