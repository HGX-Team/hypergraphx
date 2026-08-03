import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize_array(u: np.array, axis: int) -> np.array:
    """Return the normalized array u over a given axis.
    E.g., if u is a matrix NxK and axis=1, then this function returns the matrix u normalized by row.

    Parameters
    ----------
    u: numpy array.
    axis: axis along which the normalization is performed.
    """
    den1 = u.sum(axis=axis, keepdims=True)
    nzz = den1 == 0.0
    den1[nzz] = 1.0
    return u / den1


def calculate_permutation_matrix(u_ref: np.ndarray, u_pred: np.ndarray) -> np.ndarray:
    """Calculate the permutation matrix to overcome the column switching between two matrices
    using the Hungarian algorithm (linear sum assignment) for global similarity maximization.

    Parameters
    ----------
    u_ref: reference matrix.
    u_pred: matrix to switch.

    Returns
    -------
    P: permutation matrix of the same dimension as u_ref.
    """
    # Check the matrices have the same shape.
    if u_ref.shape != u_pred.shape:
        msg = f"u_ref and u_pred must have the same shape!"
        raise ValueError(msg)

    N, RANK = u_ref.shape
    M = np.dot(np.transpose(u_pred), u_ref) / float(N)  # dim = RANK x RANK

    # linear_sum_assignment minimizes cost, so negate M to maximize total similarity
    row_ind, col_ind = linear_sum_assignment(-M)

    # Build the binary permutation matrix
    P = np.zeros((RANK, RANK), dtype=float)
    P[row_ind, col_ind] = 1.0

    return P
