from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import factorial

from hnx.core import Hypergraph
from hnx.linalg import temporal_adjacency_matrix_by_order, annealed_adjacency_matrix_by_order

def intra_order_correlation_matrix_by_order(
    temporal_hypergraph: Dict[int, Hypergraph], order: int, tau: int
) -> sparse.csc_array:
    """ Compute the intra-order correlation matrix for hyperedges of order d and time lag tau.

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
    temporal_adjacency_matrix = temporal_adjacency_matrix_by_order(temporal_hypergraph, order)
    annealed_adjacency_matrix = annealed_adjacency_matrix_by_order(temporal_adjacency_matrix, order)

    correlation_matrix = sparse.csc_array(annealed_adjacency_matrix.shape, dtype=np.int8)
    for t in range(T-tau):
        adjacency_matrix_t = temporal_adjacency_matrix[t]
        adjacency_matrix_t_lagged = temporal_adjacency_matrix[t+tau]   

        centered_adjacency_matrix_t = adjacency_matrix_t - annealed_adjacency_matrix
        centered_adjacency_matrix_t_lagged = adjacency_matrix_t_lagged - annealed_adjacency_matrix

        correlation_matrix = correlation_matrix + centered_adjacency_matrix_t.dot(centered_adjacency_matrix_t_lagged.tranpose())
        
    return correlation_matrix

def intra_order_correlation_function_by_order(
    temporal_hypergraph: Dict[int, Hypergraph], order: int, tau: int
) -> float:
    """ Compute the intra-order correlation function for hyperedges of order d and time lag tau.

    Parameters
    ----------
    temporal_hypergraph: a dictionary {time : Hypergraph}.
    order: the order.
    tau: the temporal lag.

    Returns
    -------
    The intra-order correlation function of order d at time lag tau.
    """
    correlation_matrix = intra_order_correlation_matrix_by_order(temporal_hypergraph, order, tau)
    correlation_function = correlation_matrix.trace()

    return correlation_function
