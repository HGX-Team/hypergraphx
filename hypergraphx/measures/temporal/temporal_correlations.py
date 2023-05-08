from typing import Dict

import numpy as np
from scipy import sparse
from scipy.special import factorial

from hypergraphx import Hypergraph
from hypergraphx.linalg import (
    annealed_adjacency_matrix,
    temporal_adjacency_matrix_by_order,
)


def intra_order_correlation_matrix_by_order(
    temporal_hypergraph: Dict[int, Hypergraph], order: int, tau: int
) -> sparse.csc_array:
    """Compute the intra-order correlation matrix for hyperedges of order d and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order: the order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation matrix of order d at time lag tau, as a sparse matrix.
    """
    T = max(temporal_hypergraph.keys())
    temporal_adjacency_matrix = temporal_adjacency_matrix_by_order(
        temporal_hypergraph, order
    )
    annealed_adjacency_mtx = annealed_adjacency_matrix(temporal_adjacency_matrix)

    correlation_matrix = sparse.csc_array(
        annealed_adjacency_matrix.shape, dtype=np.int8
    )
    for t in range(T - tau):
        adjacency_matrix_t = temporal_adjacency_matrix[t]
        adjacency_matrix_t_lagged = temporal_adjacency_matrix[t + tau]

        centered_adjacency_matrix_t = adjacency_matrix_t - annealed_adjacency_mtx
        centered_adjacency_matrix_t_lagged = (
            adjacency_matrix_t_lagged - annealed_adjacency_mtx
        )

        correlation_matrix = correlation_matrix + centered_adjacency_matrix_t.dot(
            centered_adjacency_matrix_t_lagged.tranpose()
        )

    correlation_matrix = correlation_matrix / (factorial(order) ** 2) / (T - tau)

    return correlation_matrix


def intra_order_correlation_function_by_order(
    temporal_hypergraph: Dict[int, Hypergraph], order: int, tau: int
) -> float:
    """Compute the intra-order correlation function for hyperedges of order d and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order: the order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation function of order d at time lag tau.
    """
    correlation_matrix = intra_order_correlation_matrix_by_order(
        temporal_hypergraph, order, tau
    )
    correlation_function = correlation_matrix.trace()

    return correlation_function


def intra_order_correlation_matrices_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int, tau: int
):  # -> Tuple[sparse.csc_array]: ### to check
    """Compute the intra-order correlation matrices for hyperedges of all orders and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation matrices for all orders at time lag tau, as a dictionary {order : sparse matrix}.
    """

    correlation_matrices = {}
    for order in range(max_order):
        correlation_matrix = intra_order_correlation_matrix_by_order(
            temporal_hypergraph, order, tau
        )
        correlation_matrices[order + 1] = correlation_matrix

    return correlation_matrices


def intra_order_correlation_functions_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int, tau: int
):  # -> Tuple[float]: ### to check
    """Compute the intra-order correlation function for hyperedges of every order and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation functions for all orders at time lag tau, as a dictionary {order : function}.
    """
    correlation_functions = {}
    for order in range(max_order):
        correlation_function = intra_order_correlation_function_by_order(
            temporal_hypergraph, order, tau
        )
        correlation_functions[order + 1] = correlation_function

    return correlation_functions


def cross_order_correlation_matrix_two_orders(
    temporal_hypergraph: Dict[int, Hypergraph], order1: int, order2: int, tau: int
) -> sparse.csc_array:
    """Compute the cross-order correlation matrix between hyperedges of orders d1 and d2, and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order1: the first order.
    order2: the second order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation matrix between orders d1 and d2 at time lag tau, as a sparse matrix.
    """
    if order1 == order2:
        return intra_order_correlation_matrix_by_order(temporal_hypergraph, order1, tau)

    T = max(temporal_hypergraph.keys())
    temporal_adjacency_matrix_d1 = temporal_adjacency_matrix_by_order(
        temporal_hypergraph, order1
    )
    temporal_adjacency_matrix_d2 = temporal_adjacency_matrix_by_order(
        temporal_hypergraph, order2
    )
    annealed_adjacency_matrix_d1 = annealed_adjacency_matrix(
        temporal_adjacency_matrix_d1
    )
    annealed_adjacency_matrix_d2 = annealed_adjacency_matrix(
        temporal_adjacency_matrix_d2
    )

    correlation_matrix = sparse.csc_array(
        annealed_adjacency_matrix_d1.shape, dtype=np.int8
    )
    for t in range(T - tau):
        adjacency_matrix_d1_t = temporal_adjacency_matrix_d1[t]
        adjacency_matrix_d2_t_lagged = temporal_adjacency_matrix_d2[t + tau]

        centered_adjacency_matrix_d1_t = (
            adjacency_matrix_d1_t - annealed_adjacency_matrix_d1
        )
        centered_adjacency_matrix_d2_t_lagged = (
            adjacency_matrix_d2_t_lagged - annealed_adjacency_matrix_d2
        )

        correlation_matrix = correlation_matrix + centered_adjacency_matrix_d1_t.dot(
            centered_adjacency_matrix_d2_t_lagged.tranpose()
        )

    correlation_matrix = (
        correlation_matrix / (factorial(order1) * factorial(order2)) / (T - tau)
    )

    return correlation_matrix


def cross_order_correlation_function_two_orders(
    temporal_hypergraph: Dict[int, Hypergraph], order1: int, order2: int, tau: int
) -> float:
    """Compute the cross-order correlation function between hyperedges of order d1 and d2, at time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order1: the first order.
    order2: the second order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation function between orders d1 and d2, at time lag tau.
    """
    correlation_matrix = cross_order_correlation_matrix_two_orders(
        temporal_hypergraph, order1, order2, tau
    )
    correlation_function = correlation_matrix.trace()

    return correlation_function


def cross_order_correlation_matrices_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int, tau: int
):  # -> Tuple[sparse.csc_array]: ### to check
    """Compute the cross-order correlation matrices between each couple of hyperedge orders, time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation matrices between each couple of orders, at time lag tau,
    as a dictionary {(d1, d2) : sparse matrix}.
    """

    correlation_matrices = {}
    for order1 in range(max_order - 1):
        for order2 in range(order1, max_order):
            correlation_matrix = cross_order_correlation_matrix_two_orders(
                temporal_hypergraph, order1, order2, tau
            )
            correlation_matrices[(order1 + 1, order2 + 1)] = correlation_matrix
            if not order1 == order2:
                correlation_matrix = cross_order_correlation_matrix_two_orders(
                    temporal_hypergraph, order2, order1, tau
                )
                correlation_matrices[(order2 + 1, order1 + 1)] = correlation_matrix

    return correlation_matrices


def cross_order_correlation_functions_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int, tau: int
) -> float:
    """Compute the cross-order correlation functions between each couple of orders, at time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation functions between each couple of orders, at time lag tau,
    as a dictionary {(d1, d2 : function)}
    """
    correlation_functions = {}
    for order1 in range(max_order - 1):
        for order2 in range(order1, max_order):
            correlation_function = cross_order_correlation_function_two_orders(
                temporal_hypergraph, order1, order2, tau
            )
            correlation_functions[(order1 + 1, order2 + 1)] = correlation_function
            if not order1 == order2:
                correlation_function = cross_order_correlation_function_two_orders(
                    temporal_hypergraph, order1, order2, tau
                )
                correlation_functions[(order2 + 1, order1 + 1)] = correlation_function

    return correlation_functions


def cross_order_gap_function_two_orders(
    temporal_hypergraph: Dict[int, Hypergraph], order1: int, order2: int, tau: int
) -> float:
    """Compute the cross-order gap function between hyperedges of order d1 and d2, at time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order1: the first order.
    order2: the second order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order gap function between orders d1 and d2, at time lag tau.
    """
    sigma_d1 = intra_order_correlation_function_by_order(
        temporal_hypergraph, order1, tau=0
    )
    sigma_d2 = intra_order_correlation_function_by_order(
        temporal_hypergraph, order2, tau=0
    )
    normalization = 2 * np.sqrt(sigma_d1 * sigma_d2)
    cross_order_correlation_function_d1_d2 = (
        cross_order_correlation_function_two_orders(
            temporal_hypergraph, order1, order2, tau
        )
    )
    cross_order_correlation_function_d2_d1 = (
        cross_order_correlation_function_two_orders(
            temporal_hypergraph, order2, order1, tau
        )
    )

    cross_order_gap = (
        cross_order_correlation_function_d1_d2 - cross_order_correlation_function_d2_d1
    ) / normalization
    return cross_order_gap


def cross_order_gap_functions_all_orders(
    temporal_hypergraph: Dict[int, Hypergraph], max_order: int, tau: int
) -> float:
    """Compute the cross-order gap functions between each couple of orders, at time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order gap functions between each couple of orders, at time lag tau,
    as a dictionary {(d1, d2 : function)}
    """
    gap_functions = {}
    for order1 in range(max_order - 1):
        for order2 in range(order1, max_order):
            gap_function = cross_order_gap_function_two_orders(
                temporal_hypergraph, order1, order2, tau
            )
            gap_functions[(order1 + 1, order2 + 1)] = gap_function
            if not order1 == order2:
                correlation_function = cross_order_gap_function_two_orders(
                    temporal_hypergraph, order1, order2, tau
                )
                gap_functions[(order2 + 1, order1 + 1)] = gap_function

    return gap_functions
