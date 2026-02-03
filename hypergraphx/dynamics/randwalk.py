import numpy as np
from scipy import sparse

from hypergraphx import Hypergraph


def transition_matrix(HG: Hypergraph) -> sparse.spmatrix:
    """Compute the transition matrix of the random walk on the hypergraph.

    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.

    Returns
    -------
    K : np.ndarray

    The transition matrix of the random walk on the hypergraph.

    References
    ----------
    [1] Timoteo Carletti, Federico Battiston, Giulia Cencetti, and Duccio Fanelli, Random walks on hypergraphs, Phys. Rev. E 96, 012308 (2017)
    """
    if not HG.is_connected():
        raise ValueError("The hypergraph is not connected")

    # Build a sparse transition matrix using the binary incidence matrix B.
    # For each hyperedge e of size s, contribute (s-1) to every off-diagonal pair in e.
    # This corresponds to: M = B * diag(s_e - 1) * B^T, then zero the diagonal, then row-normalize.
    B, idx_to_node = HG.binary_incidence_matrix(return_mapping=True)
    _ = idx_to_node  # mapping is relevant for callers; matrix is in index space.

    edges = HG.get_edges()
    if len(edges) == 0:
        raise ValueError("Cannot compute a random walk on an empty hypergraph.")

    w = np.asarray([len(e) - 1 for e in edges], dtype=float)
    if np.any(w < 0):
        raise ValueError("Invalid hyperedge size encountered.")

    M = B @ sparse.diags(w, offsets=0, format="csr") @ B.T
    M = M.tocsr()
    M.setdiag(0)
    M.eliminate_zeros()

    row_sums = np.asarray(M.sum(axis=1)).ravel()
    if np.any(row_sums == 0):
        # This can happen with isolated nodes (or empty edges), which contradicts connectivity anyway.
        raise ValueError(
            "Random-walk transition undefined: a node has no outgoing probability mass."
        )

    inv_row = sparse.diags(1.0 / row_sums, offsets=0, format="csr")
    T = (inv_row @ M).tocsr()
    return T


def random_walk(
    HG: Hypergraph,
    s,
    time: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list:
    """Compute the random walk on the hypergraph.

    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.
    s : int
        The starting node of the random walk.
    time : int
        The number of steps of the random walk.
    seed : int, optional (keyword-only)
        Seed for reproducibility (does not touch global RNG state).
    rng : numpy.random.Generator, optional (keyword-only)
        Random number generator to use. If provided, `seed` must be None.

    Returns
    -------
    nodes : list
        The list of nodes visited by the random walk.
    """
    return _random_walk_impl(HG, s, time, seed=seed, rng=rng)


def _random_walk_impl(
    HG: Hypergraph,
    s,
    time: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list:
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    if time < 0:
        raise ValueError("time must be non-negative.")

    T, mapping = HG.binary_incidence_matrix(return_mapping=True)
    node_to_idx = {node: idx for idx, node in mapping.items()}
    if s not in node_to_idx:
        raise ValueError("Starting node is not in the hypergraph.")
    start_idx = node_to_idx[s]

    P = transition_matrix(HG).tocsr()
    path_idx = [start_idx]
    for _ in range(time):
        cur = path_idx[-1]
        row = P.getrow(cur)
        if row.nnz == 0:
            # This should not happen for connected hypergraphs, but keep a safe fallback.
            path_idx.append(cur)
            continue
        next_idx = int(rng.choice(row.indices, p=row.data))
        path_idx.append(next_idx)

    return [mapping[i] for i in path_idx]


def RW_stationary_state(
    HG: Hypergraph, *, tol: float = 1e-12, max_iter: int = 10000
) -> np.ndarray:
    """Compute the stationary state of the random walk on the hypergraph.

    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.

    Returns
    -------
    stationary_state : np.ndarray
        The stationary state of the random walk on the hypergraph.
    """
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    P = transition_matrix(HG).tocsr()
    n = P.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)

    # Power iteration on row-stochastic P: pi_{k+1} = pi_k P.
    for _ in range(max_iter):
        pi_next = pi @ P
        # L1 distance is natural for distributions.
        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            pi = pi_next
            break
        pi = pi_next

    total = pi.sum()
    if total == 0 or not np.isfinite(total):
        raise RuntimeError("Failed to compute a valid stationary distribution.")
    pi = pi / total
    return np.asarray(pi).ravel()


def random_walk_density(HG: Hypergraph, s: np.ndarray, time: int) -> list:
    """Compute the random walk on the hypergraph with starting density vector.

    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.
    s : np.ndarray
        The starting density vector of the random walk.

    Returns
    -------
    nodes : list
        The list of density vectors over time.
    """
    if not np.isclose(np.sum(s), 1):
        raise ValueError("The vector is not a probability density")
    if time < 0:
        raise ValueError("time must be non-negative.")

    P = transition_matrix(HG).tocsr()
    starting_density = np.asarray(s, dtype=float)
    density_list = [starting_density]
    for _ in range(time):
        starting_density = starting_density @ P
        density_list.append(np.asarray(starting_density).ravel())
    return density_list
