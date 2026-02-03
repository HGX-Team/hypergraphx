import logging
import numpy as np


def power_method(
    W,
    max_iter=1000,
    tol=1e-7,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    x0=None,
):
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    # initialize x
    if x0 is None:
        x = rng.random(len(W))
    else:
        x = np.asarray(x0, dtype=float)
    x = x / np.linalg.norm(x)
    # initialize the residual
    res = np.inf
    # initialize the number of iterations
    k = 0
    while res > tol and k < max_iter:
        # compute y
        y = np.dot(W, x)
        # compute the norm of y
        y_norm = np.linalg.norm(y)
        # compute the residual
        res = np.linalg.norm(x - y / y_norm)
        # update x
        x = y / y_norm
        # update the number of iterations
        k += 1
    return x


def CEC_centrality(HG, tol=1e-7, max_iter=1000, *, seed=None, rng=None):
    """
    Compute the CEC centrality for uniform hypergraphs.

    Parameters
    ----------

    HG : Hypergraph
        The uniform hypergraph on which the CEC centrality is computed.
    tol : float
        The tolerance for calculating the dominant eigenvalue by power method.
    max_iter : int
        The maximum number of iterations for calculating the dominant eigenvalue by power method.

    Returns
    -------
    cec : dict
        The dictionary of keys nodes of HG and values the CEC centrality of the node.


    References
    ----------
    Three Hypergraph Eigenvector Centralities,
    Austin R. Benson,
    https://doi.org/10.1137/18M1203031

    """

    # check if the hypergraph is uniform, use raise exception
    if not HG.is_uniform():
        raise Exception("The hypergraph is not uniform.")
    # check if HG is connected, use raise exception
    if not HG.is_connected():
        raise Exception("The hypergraph is not connected.")
    # define W, matrix N x N where i,j is the number of common edges between i and j
    W = np.zeros((HG.num_nodes(), HG.num_nodes()))
    order = len(HG.get_edges()[0])
    for edge in HG.get_edges():
        for i in range(order):
            for j in range(i + 1, order):
                W[edge[i], edge[j]] += 1
                W[edge[j], edge[i]] += 1
    dominant_eig = power_method(W, tol=tol, max_iter=max_iter, seed=seed, rng=rng)
    return {node: dominant_eig[node] for node in range(HG.num_nodes())}


def ZEC_centrality(HG, max_iter=1000, tol=1e-7, *, seed=None, rng=None):
    """
    Compute the ZEC centrality for uniform hypergraphs.

    Parameters
    ----------

    HG : Hypergraph
        The uniform hypergraph on which the ZEC centrality is computed.
    max_iter : int
        The maximum number of iterations.
    tol : float
        The tolerance for the stopping criterion.

    Returns
    -------
    ZEC : dict
        The dictionary of keys nodes of HG and values the ZEC centrality of the node.

    References
    ----------
    Three Hypergraph Eigenvector Centralities,
    Austin R. Benson,
    https://doi.org/10.1137/18M1203031

    """
    if not HG.is_uniform():
        raise Exception("The hypergraph is not uniform.")

    if not HG.is_connected():
        raise Exception("The hypergraph is not connected.")

    g = lambda v, e: np.prod(v[list(e)])

    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    x = rng.uniform(size=(HG.num_nodes()))
    x = x / np.linalg.norm(x, 1)

    for iter in range(max_iter):
        new_x = apply(HG, x, g)
        # multiply by the sign to try and enforce positivity
        new_x = np.sign(new_x[0]) * new_x / np.linalg.norm(new_x, 1)
        if np.linalg.norm(x - new_x) <= tol:
            break
        x = new_x.copy()
    else:
        "Iteration did not converge!"
    return {node: x[node] for node in range(HG.num_nodes())}


def HEC_centrality(HG, max_iter=100, tol=1e-6, *, seed=None, rng=None):
    """

    Compute the HEC centrality for uniform hypergraphs.

    Parameters
    ----------

    HG : Hypergraph
        The uniform hypergraph on which the HEC centrality is computed.
    max_iter : int
        The maximum number of iterations.
    tol : float
        The tolerance for the stopping criterion.

    Returns
    -------
    HEC : dict
        The dictionary of keys nodes of HG and values the HEC centrality of the node.

    References
    ----------
    Three Hypergraph Eigenvector Centralities,
    Austin R. Benson,
    https://doi.org/10.1137/18M1203031

    """
    # check if the hypergraph is uniform, use raise exception
    if not HG.is_uniform():
        raise Exception("The hypergraph is not uniform.")

    if not HG.is_connected():
        raise Exception("The hypergraph is not connected.")

    order = len(HG.get_edges()[0]) - 1
    f = lambda v, m: np.power(v, 1.0 / m)
    g = lambda v, x: np.prod(v[list(x)])

    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    x = rng.uniform(size=(HG.num_nodes()))
    x = x / np.linalg.norm(x, 1)

    for iter in range(max_iter):
        new_x = apply(HG, x, g)
        new_x = f(new_x, order)
        # Multiply by the sign to try and enforce positivity.
        new_x = np.sign(new_x[0]) * new_x / np.linalg.norm(new_x, 1)
        if np.linalg.norm(x - new_x) <= tol:
            break
        x = new_x.copy()
    else:
        logging.getLogger(__name__).warning("Iteration did not converge!")
    return {node: x[node] for node in range(HG.num_nodes())}


def apply(HG, x, g=lambda v, e: np.sum(v[list(e)])):
    new_x = np.zeros(HG.num_nodes())
    for edge in HG.get_edges():
        edge = list(edge)
        # Ordered permutations.
        for shift in range(len(edge)):
            new_x[edge[shift]] += g(x, edge[shift + 1 :] + edge[:shift])
    return new_x
