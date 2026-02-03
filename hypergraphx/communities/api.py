from __future__ import annotations

from typing import Any, Optional

import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.communities.results import (
    CorePeripheryResult,
    HypergraphMTResult,
    HyperlinkCommunitiesResult,
    HyMMSBMResult,
    HySCResult,
    hard_labels_from_memberships,
)


def run_core_periphery(
    hypergraph: Hypergraph,
    *,
    greedy_start: bool = False,
    n_iter: int = 1000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> CorePeripheryResult:
    """
    Core-periphery coreness scores.

    Returns
    -------
    CorePeripheryResult
        `scores`: dict mapping node -> coreness score.
    """
    from hypergraphx.communities.core_periphery.model import core_periphery

    scores = core_periphery(
        hypergraph,
        greedy_start=greedy_start,
        N_ITER=n_iter,
        seed=seed,
        rng=rng,
    )
    return CorePeripheryResult(scores=scores)


def run_hyperlink_communities(
    hypergraph: Hypergraph,
    *,
    load_distances: str | None = None,
    save_distances: str | None = None,
) -> HyperlinkCommunitiesResult:
    """
    Hyperlink communities (hierarchical clustering over edge distances).

    Returns
    -------
    HyperlinkCommunitiesResult
        `dendrogram`: SciPy hierarchical clustering dendrogram array.
    """
    from hypergraphx.communities.hyperlink_comm.hyperlink_communities import (
        hyperlink_communities,
    )

    dendrogram = hyperlink_communities(
        hypergraph, load_distances=load_distances, save_distances=save_distances
    )
    return HyperlinkCommunitiesResult(dendrogram=dendrogram)


def fit_hysc(
    hypergraph: Hypergraph,
    *,
    k: int,
    seed: int = 0,
    weighted_laplacian: bool = False,
    out_inference: bool = False,
    out_folder: str = "../data/output/",
    end_file: str = "_sc.dat",
) -> HySCResult:
    """
    Hypergraph Spectral Clustering (HySC).

    Returns
    -------
    HySCResult
        `memberships`: hard membership matrix u (N x K)
        `labels`: hard labels (N,)
    """
    from hypergraphx.communities.hy_sc.model import HySC

    model = HySC(
        seed=seed, out_inference=out_inference, out_folder=out_folder, end_file=end_file
    )
    memberships = model.fit(hypergraph, K=k, weighted_L=weighted_laplacian)
    labels = hard_labels_from_memberships(np.asarray(memberships))
    return HySCResult(memberships=np.asarray(memberships), labels=labels, model=model)


def fit_hypergraph_mt(
    hypergraph: Hypergraph,
    *,
    k: int,
    seed: int | None = None,
    normalize_u: bool = False,
    baseline_r0: bool = True,
    **params: Any,
) -> HypergraphMTResult:
    """
    Hypergraph-MT mixed-membership inference.

    Returns
    -------
    HypergraphMTResult
        `memberships`: membership matrix u (N x K)
        `affinity`: model affinity parameters w (shape depends on implementation)
        `max_loglik`: best achieved log-likelihood
    """
    from hypergraphx.communities.hypergraph_mt.model import HypergraphMT

    model = HypergraphMT(**params)
    memberships, affinity, max_loglik = model.fit(
        hypergraph,
        K=k,
        seed=seed,
        normalizeU=normalize_u,
        baseline_r0=baseline_r0,
    )
    memberships = np.asarray(memberships)
    labels = (
        hard_labels_from_memberships(memberships) if memberships.ndim == 2 else None
    )
    return HypergraphMTResult(
        memberships=memberships,
        affinity=np.asarray(affinity),
        max_loglik=float(max_loglik),
        labels=labels,
        model=model,
    )


def fit_hy_mmsbm(
    hypergraph: Hypergraph,
    *,
    k: int,
    seed: int | None = None,
    n_iter: int = 500,
    tol: float | None = None,
    check_convergence_every: int = 10,
    **init_params: Any,
) -> HyMMSBMResult:
    """
    Hy-MMSBM Expectation-Maximization inference.

    Returns
    -------
    HyMMSBMResult
        `memberships`: soft assignments u (N x K)
        `affinity`: affinity matrix w (K x K)
        `labels`: argmax hard labels (N,)
    """
    from hypergraphx.communities.hy_mmsbm.model import HyMMSBM

    model = HyMMSBM(K=k, seed=seed, **init_params)
    model.fit(
        hypergraph,
        n_iter=n_iter,
        tolerance=tol,
        check_convergence_every=check_convergence_every,
    )
    if model.u is None or model.w is None:
        raise RuntimeError("HyMMSBM.fit() did not produce u/w parameters.")
    memberships = np.asarray(model.u)
    labels = hard_labels_from_memberships(memberships)
    return HyMMSBMResult(
        memberships=memberships,
        affinity=np.asarray(model.w),
        trained=bool(getattr(model, "trained", True)),
        labels=labels,
        model=model,
    )
