"""Various linear operations.
In the function names, the following abbreviations are utilized:
- bf = bilinear form
- qf = quadratic form
"""

import numpy as np


def qf(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Quadratic form between a vector u and a matrix w.
    .. math::
        u^T w u

    Parameters
    ----------
    u: array of shape (..., K).
        The quadratic form operation is batched along all the dimensions but the last.
    w: square matrix of shape (K, K)

    Returns
    -------
    The array of quadratic form values. If u has shape (..., K), the returned array has
    shape (...).
    """
    K = u.shape[-1]
    assert w.shape == (K, K), "Shapes of u and w are incorrect."
    return ((u @ w) * u).sum(axis=-1)


def bf(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Bilinear form between two vectors u, v and a matrix w.
    .. math::
        u^T w v

    Parameters
    ----------
    u: array of shape (N1, K) or (K,).
        The bilinear form operation is batched for the N1 vectors is u is 2-dimensional.
    v: array of shape (N2, K) or (K,).
        The bilinear form operation is batched for the N2 vectors is v is 2-dimensional.
    w: square matrix of shape (K, K)

    Returns
    -------
    The array of bilinear form values. If u and v are batches of vectors, it has shape
    (N1, N2).
    """
    assert (
        u.shape[-1] == v.shape[-1] == w.shape[0] == w.shape[1]
    ), "Shapes are not compatible."
    return (u @ w) @ v.T


def qf_and_sum(u: np.ndarray, w: np.ndarray) -> float:
    """
    Quadratic form and sum.
    For a set of vectors u_i of length K and a matrix K, compute implements
    .. math::
        \sum_i u_i^T w u_i

    Parameters
    ----------
    u: batch of vectors
        A number N of vectors of length K are collected along the first dimension u,
        which has shape (N, K).
    w: square matrix
        Matrix of shape (K, K).

    Returns
    -------
    The scalar result of the operation.
    """
    assert u.shape[1] == w.shape[0] == w.shape[1], "Shapes are incorrect."
    return ((u @ w) * u).sum()


def bf_and_sum(u: np.ndarray, w: np.ndarray) -> float:
    """
    Bilinear form and sum.
    For a set of vectors u_i of length K and a symmetric matrix w of shape (K, K), this
    function implements
    .. math::
        \sum_{i<j} u_i^T w u_j

    Parameters
    ----------
    u: batch of vector
        A number N of vectors of length K are collected along the first dimension.
        The shape of u is (N, K).
    w: symmetric square matrix
        Matrix of shape (K, K)

    Returns
    -------
    The scalar result of the operation.
    """
    assert u.shape[1] == w.shape[0] == w.shape[1], "Shapes are incorrect."

    u_sum = u.sum(axis=0)
    return 0.5 * (qf(u_sum, w) - qf_and_sum(u, w))
