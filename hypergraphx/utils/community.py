import numpy as np


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


def calculate_permutation_matrix(u_ref: np.array, u_pred: np.array) -> np.array:
    """Calculate the permutation matrix to overcome the column switching between two matrices.

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
    rows = np.zeros(RANK)
    columns = np.zeros(RANK)
    P = np.zeros((RANK, RANK))  # permutation matrix
    for t in range(RANK):
        # Find the max element in the remaining sub-matrix,
        # the one with rows and columns removed from previous iterations
        max_entry = 0.0
        c_index = 0
        r_index = 0
        for i in range(RANK):
            if columns[i] == 0:
                for j in range(RANK):
                    if rows[j] == 0:
                        if M[j, i] > max_entry:
                            max_entry = M[j, i]
                            c_index = i
                            r_index = j
        if max_entry > 0:
            P[r_index, c_index] = 1
            columns[c_index] = 1
            rows[r_index] = 1
    if (np.sum(P, axis=1) == 0).any():
        row = np.where(np.sum(P, axis=1) == 0)[0]
        if (np.sum(P, axis=0) == 0).any():
            col = np.where(np.sum(P, axis=0) == 0)[0]
            P[row, col] = 1
    return P
