from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def hard_labels_from_memberships(u: np.ndarray) -> np.ndarray:
    """
    Convert a membership matrix `u` (N x K) to hard labels via argmax.
    """
    if u.ndim != 2:
        raise ValueError("u must be a 2D array of shape (N, K).")
    if u.shape[1] == 0:
        raise ValueError("u must have K>0 columns.")
    return np.asarray(np.argmax(u, axis=1), dtype=int)


@dataclass(frozen=True)
class CorePeripheryResult:
    """Result for core-periphery scoring.

    Attributes
    ----------
    scores : dict
        Mapping `node -> coreness score` (float).
    """

    scores: Dict[Any, float]


@dataclass(frozen=True)
class HyperlinkCommunitiesResult:
    """Result for hyperlink communities.

    Attributes
    ----------
    dendrogram : np.ndarray
        SciPy hierarchical clustering dendrogram.
        Use a cut height to extract flat edge-cluster labels.
    """

    dendrogram: np.ndarray


@dataclass(frozen=True)
class HySCResult:
    """Result for Hypergraph Spectral Clustering (HySC).

    Attributes
    ----------
    memberships : np.ndarray
        Hard-membership matrix `u` of shape (N, K).
    labels : np.ndarray
        Hard labels of shape (N,), derived from memberships.
    model : object
        The fitted HySC model instance.
    """

    memberships: np.ndarray
    labels: np.ndarray
    model: Any


@dataclass(frozen=True)
class HypergraphMTResult:
    """Result for Hypergraph-MT.

    Attributes
    ----------
    memberships : np.ndarray
        Membership matrix `u` of shape (N, K).
    affinity : np.ndarray
        Affinity parameters `w` as returned by the model.
    max_loglik : float
        Best achieved log-likelihood across realizations.
    labels : np.ndarray | None
        Optional hard labels derived from memberships (argmax). Present when the
        returned `memberships` has shape (N, K) with K>0.
    model : object
        The fitted HypergraphMT model instance.
    """

    memberships: np.ndarray
    affinity: np.ndarray
    max_loglik: float
    labels: Optional[np.ndarray]
    model: Any


@dataclass(frozen=True)
class HyMMSBMResult:
    """Result for Hy-MMSBM.

    Attributes
    ----------
    memberships : np.ndarray
        Soft assignments `u` of shape (N, K).
    affinity : np.ndarray
        Affinity matrix `w` of shape (K, K).
    trained : bool
        Whether the model reports itself as trained.
    labels : np.ndarray
        Hard labels derived from memberships (argmax).
    model : object
        The fitted HyMMSBM model instance.
    """

    memberships: np.ndarray
    affinity: np.ndarray
    trained: bool
    labels: np.ndarray
    model: Any
