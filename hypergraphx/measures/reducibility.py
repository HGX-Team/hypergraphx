import itertools
import math
from collections import Counter

import numpy as np
from scipy.special import loggamma

from hypergraphx import Hypergraph
from hypergraphx.exceptions import InvalidParameterError


def reducibility(
    hg: Hypergraph, partition=None, optimization="exact", entropy_method="count"
):
    """
    Compute hypergraph reducibility and the representative set of layers.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    partition : list or dict, optional
        Node partition for multiscale reducibility.
        If list, it must align with hg.get_nodes() order.
        If dict, it must map every node to a group.
    optimization : {"exact", "greedy"}, optional
        Optimization method for selecting representatives.
    entropy_method : {"count", "project"}, optional
        Entropy computation method.

    Returns
    -------
    tuple
        (reducibility, representative_layers)

    Notes
    -----
    If you use this function in published work, please cite:
    Kirkley, Alec, Helcio Felippe, and Federico Battiston.
    "Structural Reducibility of Hypergraphs." Physical Review Letters 135.24 (2025): 247401.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.measures.reducibility import reducibility
    >>> hg = Hypergraph()
    >>> hg.add_nodes([0, 1, 2, 3])
    >>> hg.add_edges([(0, 1), (1, 2), (2, 3), (0, 1, 2)])
    >>> eta, reps = reducibility(hg, entropy_method="count", optimization="exact")
    """
    if entropy_method not in {"count", "project"}:
        raise InvalidParameterError(
            'entropy_method must be either "count" or "project".'
        )
    if optimization not in {"exact", "greedy"}:
        raise InvalidParameterError('optimization must be either "exact" or "greedy".')

    if entropy_method == "project":
        M, Q = _get_entropies_project(hg, partition)
    else:
        M, Q = _get_entropies_count(hg, partition)

    max_layer = max(list(Q.keys()))

    if optimization == "exact":
        DLs = {}
        for reps in _powerset(list(Q.keys())):
            reps = set(reps)
            if max_layer not in reps:
                continue
            H = sum(Q[r] for r in reps)
            non_reps = set(Q.keys()) - reps
            for layer in non_reps:
                higher = [r for r in reps if r > layer]
                ces = [M[r][layer] for r in higher]
                best = higher[int(np.argmin(ces))]
                H += M[best][layer]
            DLs[tuple(sorted(reps))] = H
        Rstar = min(DLs, key=DLs.get)
        Hstar = DLs[Rstar]
    else:
        Rs, Hs = [], []
        reps = set([max_layer])
        remaining = set(Q.keys()) - reps
        Rs.append(list(reps))
        Hs.append(Q[max_layer] + sum(M[max_layer][l] for l in remaining))
        while len(remaining) > 0:
            cand_layers, cand_scores = [], []
            for candidate in remaining:
                reps_tmp = reps.union(set([candidate]))
                non_reps = remaining - set([candidate])
                H_tmp = sum(Q[r] for r in reps_tmp)
                for layer in non_reps:
                    higher = [r for r in reps_tmp if r > layer]
                    ces = [M[r][layer] for r in higher]
                    best = higher[int(np.argmin(ces))]
                    H_tmp += M[best][layer]
                cand_layers.append(candidate)
                cand_scores.append(H_tmp)
            best_index = int(np.argmin(cand_scores))
            reps.add(cand_layers[best_index])
            remaining.remove(cand_layers[best_index])
            Rs.append(list(reps))
            Hs.append(cand_scores[best_index])
        best_num_reps = int(np.argmin(Hs))
        Rstar = tuple(sorted(Rs[best_num_reps]))
        Hstar = Hs[best_num_reps]

    H0 = sum(Q[l] for l in Q)
    if H0 == Q[max_layer]:
        return 1.0, Rstar
    return (H0 - Hstar) / (H0 - Q[max_layer]), Rstar


def layer_reducibility(hg: Hypergraph, partition=None, entropy_method="count"):
    """
    Compute layer-wise reducibility.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    partition : list or dict, optional
        Node partition for multiscale reducibility.
    entropy_method : {"count", "project"}, optional
        Entropy computation method.

    Returns
    -------
    dict
        Mapping layer size -> reducibility value.

    Notes
    -----
    If you use this function in published work, please cite:
    Kirkley, Alec, Helcio Felippe, and Federico Battiston.
    "Structural Reducibility of Hypergraphs." Physical Review Letters 135.24 (2025): 247401.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.measures.reducibility import layer_reducibility
    >>> hg = Hypergraph()
    >>> hg.add_nodes([0, 1, 2, 3])
    >>> hg.add_edges([(0, 1), (1, 2), (2, 3), (0, 1, 2)])
    >>> etas = layer_reducibility(hg, entropy_method="count")
    """
    if entropy_method not in {"count", "project"}:
        raise InvalidParameterError(
            'entropy_method must be either "count" or "project".'
        )
    if entropy_method == "project":
        M, Q = _get_entropies_project(hg, partition)
    else:
        M, Q = _get_entropies_count(hg, partition)

    lmax = max(l for l in Q)
    etas = {}
    for l in Q:
        Hl = Q[l]
        best_rep = None
        best_ce = np.inf
        for k in Q:
            if k > l:
                ce = M[k][l]
                if ce <= best_ce:
                    best_ce = ce
                    best_rep = k
        if l == lmax:
            etas[l] = 0.0
        elif Hl == 0:
            etas[l] = 1.0 if best_rep != l else 0.0
        else:
            etas[l] = 1.0 - best_ce / Hl
    return etas


def _logchoose(n, k):
    """Log binomial coefficient."""
    if len(str(n)) > 300:  # stable approximation for n >> k
        if (k == 0) or (k == n):
            return 0.0
        return k * math.log(n) - k * math.log(k) + k
    n, k = float(n), float(k)
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)


def _logmultiset(n, k):
    """Log multiset coefficient."""
    return _logchoose(n + k - 1, k)


def _powerset(iterable):
    """Powerset of iterable, returned as a list without the empty set."""
    items = list(iterable)
    pset = list(
        itertools.chain.from_iterable(
            itertools.combinations(items, r) for r in range(len(items) + 1)
        )
    )
    return pset[1:]


def _normalize_partition(partition, hg: Hypergraph):
    if partition is None:
        return None
    nodes = hg.get_nodes()
    if isinstance(partition, dict):
        try:
            return {node: partition[node] for node in nodes}
        except KeyError as exc:
            raise InvalidParameterError(
                "Partition dict must include every node in the hypergraph."
            ) from exc
    if len(partition) != len(nodes):
        raise InvalidParameterError(
            "Partition list must match the number of nodes in the hypergraph."
        )
    return {node: group for node, group in zip(nodes, partition)}


def _projection(edges, sizes):
    """
    Project a collection of hyperedges onto layers with sizes in sizes.
    Returns a dict {size: set(edges)}.
    """
    sizes = sorted(sizes)[::-1]
    projections = {}
    for size in sizes:
        projections[size] = set()
        for edge in edges:
            for combo in itertools.combinations(edge, size):
                projections[size].add(combo)
    return projections


def _get_layers(hg: Hypergraph, sizes="all"):
    """
    Return layers of hypergraph as a dict {size: set(edges)}.
    """
    edges_by_size = hg.edges_by_size(index_by="edge_key")
    if sizes == "all":
        sizes = list(edges_by_size.keys())
    layers = {}
    for size in sizes:
        layers[size] = set(edges_by_size.get(size, []))
    return layers


def _coarse_grain(edges, partition_map):
    """
    Return coarse-grained hypergraph according to node partition.
    Returns a dict {tuple: count} to represent a multiset.
    """
    coarse = {}
    for edge in edges:
        mapped = [partition_map[node] for node in edge]
        new_edge = tuple(sorted(mapped))
        coarse[new_edge] = coarse.get(new_edge, 0) + 1
    return coarse


def _all_projections(layers):
    """
    Return dict P such that P[k][l] is projection of layer k onto tuples of size l.
    """
    projections = {}
    indices = sorted(list(layers.keys()))[::-1]
    for index, k in enumerate(indices):
        projections[k] = {}
        remaining = indices[index + 1 :]
        for subindex, l in enumerate(remaining):
            if subindex == 0:
                projections[k][l] = _projection(layers[k], [l])[l]
            else:
                prev = remaining[subindex - 1]
                projections[k][l] = _projection(projections[k][prev], [l])[l]
    return projections


def _get_entropies_project(hg: Hypergraph, partition=None):
    """
    Returns:
        M: dict such that M[k][l] is conditional entropy of higher layer k to lower layer l.
        Q: dict such that Q[l] is entropy of layer l.
    """
    if not isinstance(hg, Hypergraph):
        raise InvalidParameterError("Reducibility supports undirected Hypergraph only.")

    partition_map = _normalize_partition(partition, hg)
    if partition_map is not None:
        num_blocks = len(set(partition_map.values()))

    num_nodes = hg.num_nodes()

    layers = _get_layers(hg)
    projections = _all_projections(layers)

    def entropy(size):
        edge_count = len(layers[size])
        if partition_map is not None:
            return _logmultiset(math.comb(num_blocks + size - 1, size), edge_count)
        return _logchoose(math.comb(num_nodes, size), edge_count)

    def conditional_entropy(k, l):
        edge_count = len(layers[l])
        proj_size = len(projections[k][l])
        if partition_map is not None:
            proj_coarse = _coarse_grain(projections[k][l], partition_map)
            lower_coarse = _coarse_grain(layers[l], partition_map)
            overlap = 0
            for edge in lower_coarse:
                if edge in proj_coarse:
                    overlap += min(proj_coarse[edge], lower_coarse[edge])
            return _logchoose(proj_size, overlap) + _logmultiset(
                math.comb(num_blocks + l - 1, l), edge_count - overlap
            )
        overlap = len(projections[k][l].intersection(layers[l]))
        return _logchoose(proj_size, overlap) + _logchoose(
            math.comb(num_nodes, l) - proj_size, edge_count - overlap
        )

    M, Q = {}, {}
    for k in layers:
        M[k] = {}
        for l in layers:
            if k > l:
                M[k][l] = conditional_entropy(k, l)
        Q[k] = entropy(k)
    return M, Q


def _get_entropies_count(hg: Hypergraph, partition=None):
    """
    Returns:
        M: dict such that M[k][l] is conditional entropy of higher layer k to lower layer l.
        Q: dict such that Q[l] is entropy of layer l.
    """
    if not isinstance(hg, Hypergraph):
        raise InvalidParameterError("Reducibility supports undirected Hypergraph only.")

    layers = _get_layers(hg)
    num_nodes = hg.num_nodes()
    partition_map = _normalize_partition(partition, hg)

    if partition_map is not None:
        num_blocks = len(set(partition_map.values()))

    M, Q = {}, {}
    indices = sorted(list(layers.keys()))[::-1]
    for ii, k in enumerate(indices):
        Ek = len(layers[k])
        if partition_map is not None:
            Q[k] = _logmultiset(math.comb(num_blocks + k - 1, k), Ek)
        else:
            Q[k] = _logchoose(math.comb(num_nodes, k), Ek)

        lower_indices = indices[ii + 1 :]

        def get_sizes_proj(layer):
            sizes = Counter({l: 0 for l in lower_indices})
            if len(layer) == 0:
                return sizes
            checked = []
            layer = list(layer)
            for edge in layer:
                overlaps = set()
                for prev in checked:
                    inter = set(edge).intersection(set(prev))
                    overlaps.add(tuple(sorted(list(inter))))
                sizes_max = Counter({l: math.comb(len(edge), l) for l in lower_indices})
                sizes += sizes_max - get_sizes_proj(overlaps)
                checked.append(edge)
            return sizes

        Ek2ls = get_sizes_proj(layers[k])

        M[k] = {}
        for l in lower_indices:
            El = len(layers[l])
            Ek2l = Ek2ls[l]
            overlap = 0
            if partition_map is not None:
                lower_coarse = _coarse_grain(layers[l], partition_map)
                for lower_tup in lower_coarse:
                    lower_counts = Counter(list(lower_tup))
                    num_lower_in_higher = 0
                    for higher_tup in layers[k]:
                        higher_counts = Counter(
                            [partition_map[node] for node in higher_tup]
                        )
                        hl_counts = higher_counts & lower_counts
                        if hl_counts == lower_counts:
                            num_lower_in_higher += np.prod(
                                [
                                    math.comb(higher_counts[i], lower_counts[i])
                                    for i in lower_counts
                                ]
                            )
                    overlap += min(num_lower_in_higher, lower_coarse[lower_tup])
                M[k][l] = _logchoose(Ek2l, overlap) + _logmultiset(
                    math.comb(num_blocks + l - 1, l), El - overlap
                )
            else:
                lower_tmp = layers[l].copy()
                for higher in layers[k]:
                    higher_set = set(higher)
                    overlapping = set()
                    for lower in lower_tmp:
                        if len(set(lower).intersection(higher_set)) == l:
                            overlapping.add(lower)
                    for edge in overlapping:
                        lower_tmp.remove(edge)
                        overlap += 1
                M[k][l] = _logchoose(Ek2l, overlap) + _logchoose(
                    math.comb(num_nodes, l) - Ek2l, El - overlap
                )
    return M, Q


def reducibility(
    hg: Hypergraph, partition=None, optimization="exact", entropy_method="count"
):
    """
    Compute hypergraph reducibility and the representative set of layers.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    partition : list or dict, optional
        Node partition for multiscale reducibility.
        If list, it must align with hg.get_nodes() order.
        If dict, it must map every node to a group.
    optimization : {"exact", "greedy"}, optional
        Optimization method for selecting representatives.
    entropy_method : {"count", "project"}, optional
        Entropy computation method.

    Returns
    -------
    tuple
        (reducibility, representative_layers)

    Notes
    -----
    If you use this function in published work, please cite:
    Kirkley, Alec, Helcio Felippe, and Federico Battiston.
    "Structural Reducibility of Hypergraphs." Physical Review Letters 135.24 (2025): 247401.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.measures.reducibility import reducibility
    >>> hg = Hypergraph()
    >>> hg.add_nodes([0, 1, 2, 3])
    >>> hg.add_edges([(0, 1), (1, 2), (2, 3), (0, 1, 2)])
    >>> eta, reps = reducibility(hg, entropy_method="count", optimization="exact")
    """
    if entropy_method not in {"count", "project"}:
        raise InvalidParameterError(
            'entropy_method must be either "count" or "project".'
        )
    if optimization not in {"exact", "greedy"}:
        raise InvalidParameterError('optimization must be either "exact" or "greedy".')

    if entropy_method == "project":
        M, Q = _get_entropies_project(hg, partition)
    else:
        M, Q = _get_entropies_count(hg, partition)

    max_layer = max(list(Q.keys()))

    if optimization == "exact":
        DLs = {}
        for reps in _powerset(list(Q.keys())):
            reps = set(reps)
            if max_layer not in reps:
                continue
            H = sum(Q[r] for r in reps)
            non_reps = set(Q.keys()) - reps
            for layer in non_reps:
                higher = [r for r in reps if r > layer]
                ces = [M[r][layer] for r in higher]
                best = higher[int(np.argmin(ces))]
                H += M[best][layer]
            DLs[tuple(sorted(reps))] = H
        Rstar = min(DLs, key=DLs.get)
        Hstar = DLs[Rstar]
    else:
        Rs, Hs = [], []
        reps = set([max_layer])
        remaining = set(Q.keys()) - reps
        Rs.append(list(reps))
        Hs.append(Q[max_layer] + sum(M[max_layer][l] for l in remaining))
        while len(remaining) > 0:
            cand_layers, cand_scores = [], []
            for candidate in remaining:
                reps_tmp = reps.union(set([candidate]))
                non_reps = remaining - set([candidate])
                H_tmp = sum(Q[r] for r in reps_tmp)
                for layer in non_reps:
                    higher = [r for r in reps_tmp if r > layer]
                    ces = [M[r][layer] for r in higher]
                    best = higher[int(np.argmin(ces))]
                    H_tmp += M[best][layer]
                cand_layers.append(candidate)
                cand_scores.append(H_tmp)
            best_index = int(np.argmin(cand_scores))
            reps.add(cand_layers[best_index])
            remaining.remove(cand_layers[best_index])
            Rs.append(list(reps))
            Hs.append(cand_scores[best_index])
        best_num_reps = int(np.argmin(Hs))
        Rstar = tuple(sorted(Rs[best_num_reps]))
        Hstar = Hs[best_num_reps]

    H0 = sum(Q[l] for l in Q)
    if H0 == Q[max_layer]:
        return 1.0, Rstar
    return (H0 - Hstar) / (H0 - Q[max_layer]), Rstar


def layer_reducibility(hg: Hypergraph, partition=None, entropy_method="count"):
    """
    Compute layer-wise reducibility.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    partition : list or dict, optional
        Node partition for multiscale reducibility.
    entropy_method : {"count", "project"}, optional
        Entropy computation method.

    Returns
    -------
    dict
        Mapping layer size -> reducibility value.

    Notes
    -----
    If you use this function in published work, please cite:
    Kirkley, Alec, Helcio Felippe, and Federico Battiston.
    "Structural Reducibility of Hypergraphs." Physical Review Letters 135.24 (2025): 247401.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.measures.reducibility import layer_reducibility
    >>> hg = Hypergraph()
    >>> hg.add_nodes([0, 1, 2, 3])
    >>> hg.add_edges([(0, 1), (1, 2), (2, 3), (0, 1, 2)])
    >>> etas = layer_reducibility(hg, entropy_method="count")
    """
    if entropy_method not in {"count", "project"}:
        raise InvalidParameterError(
            'entropy_method must be either "count" or "project".'
        )
    if entropy_method == "project":
        M, Q = _get_entropies_project(hg, partition)
    else:
        M, Q = _get_entropies_count(hg, partition)

    lmax = max(l for l in Q)
    etas = {}
    for l in Q:
        Hl = Q[l]
        best_rep = None
        best_ce = np.inf
        for k in Q:
            if k > l:
                ce = M[k][l]
                if ce <= best_ce:
                    best_ce = ce
                    best_rep = k
        if l == lmax:
            etas[l] = 0.0
        elif Hl == 0:
            etas[l] = 1.0 if best_rep != l else 0.0
        else:
            etas[l] = 1.0 - best_ce / Hl
    return etas
