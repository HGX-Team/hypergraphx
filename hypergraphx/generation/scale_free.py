import math
import numpy as np
from hypergraphx import Hypergraph


def scale_free_hypergraph(
    num_nodes: int,
    edges_by_size: dict[int, int],
    alpha_by_size: float | dict[int, float] = 1.0,
    rho: float = 0,
    seed: int | None = None,
    enforce_unique_edges: bool = True,
    max_tries_factor: int = 50,
):
    """
    Generate a hypergraph from a hidden-variable (fitness / activity) model with
    heavy-tailed node activities, and tunable inter-size rank correlation.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (labeled 0..num_nodes-1).
    edges_by_size : dict[int, int]
        Mapping {hyperedge size (cardinality): number of hyperedges}.
    alpha_by_size : float | dict[int, float]
        Zipf exponent(s) controlling activity heterogeneity. Larger alpha => stronger hub dominance.
        If float, the same alpha is used for all sizes.
    rho : float
        Equicorrelation of latent Gaussian scores across sizes (controls inter-size rank correlation of activities).
        Can be negative, subject to rho ∈ [-1/(m-1), 1] when m>1.
    seed : int | None
        Seed for reproducibility.
    enforce_unique_edges : bool
        If True, enforces a simple hypergraph per size (no duplicate hyperedges) by rejection sampling.
    max_tries_factor : int
        Collision budget for uniqueness per size: ~ max_tries_factor * num_edges attempts.

    Returns
    -------
    Hypergraph
        Generated hypergraph.
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not edges_by_size:
        raise ValueError("edges_by_size must be a non-empty dict {size: num_edges}")

    sizes = sorted(int(s) for s in edges_by_size.keys())
    m_sizes = len(sizes)

    # Validate sizes/counts
    for s in sizes:
        ecount = edges_by_size[s]
        if int(ecount) != ecount or ecount < 0:
            raise ValueError(f"edges_by_size[{s}] must be a nonnegative int")
        if s < 1:
            raise ValueError(f"Hyperedge size must be >= 1, got {s}")
        if s > num_nodes:
            raise ValueError(f"Hyperedge size {s} cannot exceed num_nodes {num_nodes}")

        if enforce_unique_edges:
            max_unique = math.comb(num_nodes, s)
            if ecount > max_unique:
                raise ValueError(
                    f"Requested {ecount} unique hyperedges of size {s}, but only "
                    f"C({num_nodes},{s})={max_unique} exist."
                )

    # Normalize alpha_by_size
    if isinstance(alpha_by_size, dict):
        alpha_for = {int(k): float(v) for k, v in alpha_by_size.items()}
    else:
        alpha_for = {s: float(alpha_by_size) for s in sizes}

    for s in sizes:
        if s not in alpha_for:
            raise ValueError(f"Missing alpha for size {s} in alpha_by_size")
        if not (alpha_for[s] > 0):
            raise ValueError(
                f"alpha must be > 0; got alpha_by_size[{s}]={alpha_for[s]}"
            )

    # Correlation structure across size-layers
    if m_sizes > 1:
        lo = -1.0 / (m_sizes - 1)
        if not (lo <= rho <= 1.0):
            raise ValueError(
                f"For {m_sizes} sizes, rho must be in [{lo}, 1]. Got {rho}."
            )
        C = np.full((m_sizes, m_sizes), rho, dtype=float)
        np.fill_diagonal(C, 1.0)
        L = np.linalg.cholesky(C)
    else:
        L = np.array([[1.0]], dtype=float)

    rng = np.random.default_rng(seed)
    nodes = np.arange(num_nodes)

    # Latent Gaussian scores: (num_nodes × m_sizes)
    Z = rng.standard_normal((num_nodes, m_sizes))
    Y = Z @ L.T

    H = Hypergraph()
    H.add_nodes(list(range(num_nodes)))

    for j, s in enumerate(sizes):
        alpha = alpha_for[s]
        scores = Y[:, j]

        # Zipf activities from rank: w_i = rank_i^{-alpha}
        order = np.argsort(scores)[::-1]  # rank 1 = largest score
        ranks = np.empty(num_nodes, dtype=float)
        ranks[order] = np.arange(1, num_nodes + 1, dtype=float)
        w = ranks ** (-alpha)
        p = w / w.sum()

        num_edges = int(edges_by_size[s])

        if enforce_unique_edges:
            edges = set()
            tries = 0
            max_tries = max(10, max_tries_factor * max(1, num_edges))
            while len(edges) < num_edges:
                if tries > max_tries:
                    raise RuntimeError(
                        f"Too many collisions sampling unique hyperedges (size={s}). "
                        f"Reduce edges_by_size[{s}], reduce alpha, or set enforce_unique_edges=False."
                    )
                choice = rng.choice(nodes, size=s, replace=False, p=p)
                edges.add(tuple(sorted(map(int, choice))))
                tries += 1
            H.add_edges(edges)
        else:
            edges = []
            for _ in range(num_edges):
                choice = rng.choice(nodes, size=s, replace=False, p=p)
                edges.append(tuple(sorted(map(int, choice))))
            H.add_edges(edges)

    return H
