import numpy as np
from scipy.sparse import csc_array

from hypergraphx.measures.temporal.temporal_correlations import (
    intra_order_correlation_matrix_by_order,
    intra_order_correlation_function_by_order,
    intra_order_correlation_matrices_all_orders,
    intra_order_correlation_functions_all_orders,
    cross_order_correlation_matrix_two_orders,
    cross_order_correlation_function_two_orders,
)


def _make_temporal_matrices():
    m0 = csc_array([[0, 1], [1, 0]], dtype=float)
    m1 = csc_array([[0, 2], [2, 0]], dtype=float)
    adj = {1: {0: m0, 1: m1}, 2: {0: m0, 1: m1}}
    annealed = {1: (m0 + m1) / 2, 2: (m0 + m1) / 2}
    return adj, annealed


def test_intra_order_correlation_matrix_and_function():
    """Test intra-order correlation shape and trace match."""
    adj, annealed = _make_temporal_matrices()

    mat = intra_order_correlation_matrix_by_order(adj, annealed, order=1, tau=0)
    fun = intra_order_correlation_function_by_order(adj, annealed, order=1, tau=0)

    assert mat.shape == (2, 2)
    assert np.isclose(mat.trace(), fun)


def test_intra_order_correlation_all_orders():
    """Test intra-order correlation for all orders returns dicts."""
    adj, annealed = _make_temporal_matrices()

    mats = intra_order_correlation_matrices_all_orders(
        adj, annealed, max_order=2, tau=0
    )
    funcs = intra_order_correlation_functions_all_orders(
        adj, annealed, max_order=2, tau=0
    )

    assert set(mats.keys()) == {1, 2}
    assert set(funcs.keys()) == {1, 2}


def test_cross_order_correlation_matrix_and_function():
    """Test cross-order correlation works for different orders."""
    adj, annealed = _make_temporal_matrices()

    mat = cross_order_correlation_matrix_two_orders(
        adj, annealed, order1=1, order2=2, tau=0
    )
    fun = cross_order_correlation_function_two_orders(
        adj, annealed, order1=1, order2=2, tau=0
    )

    assert mat.shape == (2, 2)
    assert np.isfinite(fun)
