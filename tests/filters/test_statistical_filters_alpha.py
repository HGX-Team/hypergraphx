import pytest

from hypergraphx import Hypergraph
from hypergraphx.filters.statistical_filters import get_svh, get_svc


def test_statistical_filters_accept_integer_like_weights_and_alpha_param():
    # Weighted hypergraph with integer multiplicities (as floats) should be accepted.
    hg = Hypergraph(edge_list=[(0, 1, 2), (0, 1, 3)], weighted=True, weights=[2.0, 1.0])
    out = get_svh(hg, max_order=3, alpha=0.5, mp=False)
    assert isinstance(out, dict)

    out2 = get_svc(hg, min_order=2, max_order=3, alpha=0.5, mp=False)
    assert out2 is not None


def test_statistical_filters_reject_non_integer_weights():
    hg = Hypergraph(edge_list=[(0, 1, 2)], weighted=True, weights=[0.5])
    with pytest.raises(ValueError, match="integer edge weights"):
        _ = get_svh(hg, max_order=3, alpha=0.01, mp=False)
