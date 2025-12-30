from __future__ import annotations

from typing import List

import numpy as np
from scipy import sparse


def _row_sums(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1)
    if hasattr(row_sums, "A1"):
        return row_sums.A1
    return np.asarray(row_sums).flatten()


def isolates_from_incidence(
    incidence: sparse.spmatrix | np.ndarray,
) -> np.ndarray:
    """Return indices of isolated nodes from an incidence matrix."""
    row_sums = _row_sums(incidence)
    return np.where(row_sums == 0)[0]


def non_isolates_from_incidence(
    incidence: sparse.spmatrix | np.ndarray,
) -> np.ndarray:
    """Return indices of non-isolated nodes from an incidence matrix."""
    row_sums = _row_sums(incidence)
    return np.where(row_sums != 0)[0]


def hyperedges_per_node(incidence: sparse.csr_matrix) -> List[np.ndarray]:
    """Return indices of incident hyperedges for each node from a CSR matrix."""
    return np.split(incidence.indices, incidence.indptr)[1:-1]
