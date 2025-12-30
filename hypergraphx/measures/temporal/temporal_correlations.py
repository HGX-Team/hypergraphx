from typing import Dict

import logging
import numpy as np
from scipy import sparse
from scipy.special import factorial

from hypergraphx.linalg import (
    annealed_adjacency_matrix,
)


def intra_order_correlation_matrix_by_order(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    order=1,
    tau=0,
) -> sparse.csc_array:
    """Compute the intra-order correlation matrix for hyperedges of order d and time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    order: the order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation matrix of order d at time lag tau, as a sparse matrix.
    """
    temporal_adjacency_matrix = adjacency_matrices_all_orders[order]
    T = len(temporal_adjacency_matrix.keys())

    if annealed_adjacency_matrices_all_orders == None:
        annealed_adjacency_mtx = annealed_adjacency_matrix(temporal_adjacency_matrix)
    else:
        annealed_adjacency_mtx = annealed_adjacency_matrices_all_orders[order]

    correlation_matrix = sparse.csc_array(annealed_adjacency_mtx.shape, dtype=np.int8)
    for t in range(T - tau):
        adjacency_matrix_t = temporal_adjacency_matrix[t]
        adjacency_matrix_t_lagged = temporal_adjacency_matrix[t + tau]

        centered_adjacency_matrix_t = adjacency_matrix_t - annealed_adjacency_mtx
        centered_adjacency_matrix_t_lagged = (
            adjacency_matrix_t_lagged - annealed_adjacency_mtx
        )

        correlation_matrix = correlation_matrix + centered_adjacency_matrix_t.dot(
            centered_adjacency_matrix_t_lagged.transpose()
        )

    correlation_matrix = correlation_matrix / (factorial(order) ** 2) / (T - tau)

    return correlation_matrix


def intra_order_correlation_function_by_order(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    order=1,
    tau=0,
) -> float:
    """Compute the intra-order correlation function for hyperedges of order d and time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    order: the order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation function of order d at time lag tau.
    """
    correlation_matrix = intra_order_correlation_matrix_by_order(
        adjacency_matrices_all_orders,
        annealed_adjacency_matrices_all_orders,
        order,
        tau,
    )
    correlation_function = correlation_matrix.trace()

    return correlation_function


def intra_order_correlation_matrices_all_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    max_order=None,
    tau=0,
):  # -> Tuple[sparse.csc_array]: ### to check
    """Compute the intra-order correlation matrices for hyperedges of all orders and time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation matrices for all orders at time lag tau, as a dictionary {order : sparse matrix}.
    """

    correlation_matrices = dict()
    if max_order == None:
        max_order = max(adjacency_matrices_all_orders.keys())
    for order in range(1, max_order + 1):
        correlation_matrix = intra_order_correlation_matrix_by_order(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            order,
            tau,
        )
        correlation_matrices[order] = correlation_matrix

    return correlation_matrices


def intra_order_correlation_functions_all_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    max_order=None,
    tau=0,
):  # -> Tuple[float]: ### to check
    """Compute the intra-order correlation function for hyperedges of every order and time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation functions for all orders at time lag tau, as a dictionary {order : function}.
    """
    correlation_functions = dict()
    if max_order == None:
        max_order = max(adjacency_matrices_all_orders.keys())
    for order in range(1, max_order + 1):
        correlation_function = intra_order_correlation_function_by_order(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            order,
            tau,
        )
        correlation_functions[order] = correlation_function

    return correlation_functions


def cross_order_correlation_matrix_two_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    order1=1,
    order2=1,
    tau=0,
) -> sparse.csc_array:
    """Compute the cross-order correlation matrix between hyperedges of orders d1 and d2, and time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    order1: the first order.
    order2: the second order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation matrix between orders d1 and d2 at time lag tau, as a sparse matrix.
    """
    if order1 == order2:
        return intra_order_correlation_matrix_by_order(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            order1,
            tau,
        )

    temporal_adjacency_matrix_d1 = adjacency_matrices_all_orders[order1]
    temporal_adjacency_matrix_d2 = adjacency_matrices_all_orders[order2]
    T = len(temporal_adjacency_matrix_d1.keys())

    if annealed_adjacency_matrices_all_orders == None:
        annealed_adjacency_matrix_d1 = annealed_adjacency_matrix(
            temporal_adjacency_matrix_d1
        )
        annealed_adjacency_matrix_d2 = annealed_adjacency_matrix(
            temporal_adjacency_matrix_d2
        )
    else:
        annealed_adjacency_matrix_d1 = annealed_adjacency_matrices_all_orders[order1]
        annealed_adjacency_matrix_d2 = annealed_adjacency_matrices_all_orders[order2]

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
            centered_adjacency_matrix_d2_t_lagged.transpose()
        )

    correlation_matrix = (
        correlation_matrix / (factorial(order1) * factorial(order2)) / (T - tau)
    )

    return correlation_matrix


def cross_order_correlation_function_two_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    order1=1,
    order2=1,
    tau=0,
    normalized=False,
) -> float:
    """Compute the cross-order correlation function between hyperedges of order d1 and d2, at time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    order1: the first order.
    order2: the second order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation function between orders d1 and d2, at time lag tau.
    """
    correlation_matrix = cross_order_correlation_matrix_two_orders(
        adjacency_matrices_all_orders,
        annealed_adjacency_matrices_all_orders,
        order1,
        order2,
        tau,
    )
    correlation_function = correlation_matrix.trace()
    if normalized:
        sigma_d1 = intra_order_correlation_function_by_order(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            order1,
            tau=0,
        )
        sigma_d2 = intra_order_correlation_function_by_order(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            order2,
            tau=0,
        )
        normalization = 2 * np.sqrt(sigma_d1 * sigma_d2)

        correlation_function = correlation_function / normalization

    return correlation_function


def cross_order_correlation_matrices_all_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    max_order=None,
    tau=0,
):  # -> Tuple[sparse.csc_array]: ### to check
    """Compute the cross-order correlation matrices between each couple of hyperedge orders, time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation matrices between each couple of orders, at time lag tau,
    as a dictionary {(d1, d2) : sparse matrix}.
    """

    correlation_matrices = dict()
    if max_order == None:
        max_order = max(adjacency_matrices_all_orders.keys())
    for order1 in range(1, max_order + 1):
        for order2 in range(order1, max_order + 1):
            correlation_matrix = cross_order_correlation_matrix_two_orders(
                adjacency_matrices_all_orders,
                annealed_adjacency_matrices_all_orders,
                order1,
                order2,
                tau,
            )
            correlation_matrices[(order1, order2)] = correlation_matrix
            if not order1 == order2:
                correlation_matrix = cross_order_correlation_matrix_two_orders(
                    adjacency_matrices_all_orders,
                    annealed_adjacency_matrices_all_orders,
                    order2,
                    order1,
                    tau,
                )
                correlation_matrices[(order2, order1)] = correlation_matrix

    return correlation_matrices


def cross_order_correlation_functions_all_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    max_order=None,
    tau=0,
    normalized=False,
) -> float:
    """Compute the cross-order correlation functions between each couple of orders, at time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order correlation functions between each couple of orders, at time lag tau,
    as a dictionary {(d1, d2 : function)}
    """
    correlation_functions = dict()
    if max_order == None:
        max_order = max(adjacency_matrices_all_orders.keys())

    if normalized:
        sigmas = intra_order_correlation_functions_all_orders(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            max_order,
            tau=0,
        )

    for order1 in range(1, max_order + 1):
        for order2 in range(order1, max_order + 1):
            # order1 before order2
            correlation_function = cross_order_correlation_function_two_orders(
                adjacency_matrices_all_orders,
                annealed_adjacency_matrices_all_orders,
                order1,
                order2,
                tau,
            )
            correlation_functions[(order1, order2)] = correlation_function

            # order2 before order1
            if not order1 == order2:
                correlation_function = cross_order_correlation_function_two_orders(
                    adjacency_matrices_all_orders,
                    annealed_adjacency_matrices_all_orders,
                    order1,
                    order2,
                    tau,
                )
                correlation_functions[(order2, order1)] = correlation_function

            # normalization
            if normalized:
                normalization = 2 * np.sqrt(sigmas[order1] * sigmas[order2])
                correlation_functions[(order1, order2)] = (
                    correlation_functions[(order1, order2)] / normalization
                )
                if not order1 == order2:
                    correlation_functions[(order2, order1)] = (
                        correlation_functions[(order2, order1)] / normalization
                    )

    return correlation_functions


def cross_order_gap_function_two_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    order1=1,
    order2=1,
    tau=0,
) -> float:
    """Compute the cross-order gap function between hyperedges of order d1 and d2, at time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    order1: the first order.
    order2: the second order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order gap function between orders d1 and d2, at time lag tau.
    """
    if order1 == order2:
        logging.getLogger(__name__).warning(
            "It is not meaningful to evaluate a cross-order gap within the same order"
        )
        return 0

    sigma_d1 = intra_order_correlation_function_by_order(
        adjacency_matrices_all_orders,
        annealed_adjacency_matrices_all_orders,
        order1,
        tau=0,
    )
    sigma_d2 = intra_order_correlation_function_by_order(
        adjacency_matrices_all_orders,
        annealed_adjacency_matrices_all_orders,
        order2,
        tau=0,
    )
    normalization = 2 * np.sqrt(sigma_d1 * sigma_d2)
    cross_order_correlation_function_d1_d2 = (
        cross_order_correlation_function_two_orders(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            order1,
            order2,
            tau,
        )
    )
    cross_order_correlation_function_d2_d1 = (
        cross_order_correlation_function_two_orders(
            adjacency_matrices_all_orders,
            annealed_adjacency_matrices_all_orders,
            order2,
            order1,
            tau,
        )
    )

    cross_order_gap = (
        cross_order_correlation_function_d1_d2 - cross_order_correlation_function_d2_d1
    ) / normalization
    return cross_order_gap


def cross_order_gap_functions_all_orders(
    adjacency_matrices_all_orders: Dict[int, Dict[int, sparse.csc_array]],
    annealed_adjacency_matrices_all_orders=None,
    max_order=None,
    tau=0,
) -> float:
    """Compute the cross-order gap functions between each couple of orders, at time lag tau.

    Parameters
    ----------
    adjacency_matrices_all_orders: a dictionary {order : {time : adjacency matrix}}.
    annealed_adjacency_matrices_all_orders: a dictionary {order : annealed adjacency matrix}.
    max_order: the maximum order.
    tau: the temporal lag.

    Returns
    -------
    The cross-order gap functions between each couple of orders, at time lag tau,
    as a dictionary {(d1, d2 : function)}
    """
    gap_functions = dict()
    if max_order == None:
        max_order = max(adjacency_matrices_all_orders.keys())
    for order1 in range(1, max_order + 1):
        for order2 in range(order1, max_order + 1):
            if order1 == order2:
                gap_functions[(order1, order2)] = 0
            else:
                gap_function = cross_order_gap_function_two_orders(
                    adjacency_matrices_all_orders,
                    annealed_adjacency_matrices_all_orders,
                    order1,
                    order2,
                    tau,
                )
                gap_functions[(order1, order2)] = gap_function
                gap_functions[(order2, order1)] = -1 * gap_function

    return gap_functions
