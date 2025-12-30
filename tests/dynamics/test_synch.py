import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.dynamics.synch import higher_order_MSF


def test_higher_order_msf_returns_none_when_not_applicable():
    """Test higher_order_MSF returns None when no MSF can be evaluated."""
    hg = Hypergraph(edge_list=[(0, 1)], weighted=True, weights=[1.0])

    F = lambda t, x, *params: x
    JF = lambda x, *params: np.eye(len(x))
    JH1 = lambda x: np.eye(len(x))
    JH2 = lambda x: np.zeros((len(x), len(x)))

    result = higher_order_MSF(
        hypergraph=hg,
        dim=2,
        F=F,
        JF=JF,
        params=(),
        sigmas=[1.0, 1.0],
        JHs=[JH1, JH2],
        X0=np.array([0.1, 0.2]),
        interval=[0.1, 0.2],
        diffusive_like=True,
        integration_time=1.0,
        integration_step=0.1,
        C=1,
        verbose=False,
    )

    assert result is None
