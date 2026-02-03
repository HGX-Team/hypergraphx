import math
import random

import logging
import numpy as np

from hypergraphx import Hypergraph


def relabel_nodes(d, w):
    N = set()

    node2id = {}
    idx = 0

    new_d = set()
    new_w = {}

    for e in d:
        for n in e:
            if n not in node2id:
                node2id[n] = idx
                idx += 1
            N.add(n)

        tmp = []
        for n in e:
            tmp.append(node2id[n])
        tmp = tuple(sorted(tmp))
        new_d.add(tmp)
        new_w[tmp] = w[e]

    return len(N), node2id, new_d, new_w


def sample_params(*, rng: np.random.Generator | None = None):
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(0, 1), rng.uniform(0, 1)


def transition_function(i, N_nodes, a, b):
    if i <= math.floor(b * N_nodes):
        return (i * (1 - a)) / (2 * math.floor(b * N_nodes))
    else:
        return ((i - math.floor(b * N_nodes)) * (1 - a)) / (
            2 * (N_nodes - math.floor(b * N_nodes))
        ) + (1 + a) / 2


def aggregation_function(values):
    return sum(values)


def aggregate_local_core_values(nodes, order, local_core_values):
    values = []
    for n in nodes:
        values.append(local_core_values[order[n]])
    return aggregation_function(values)


def get_core_quality(edges, order, local_core_values):
    R = 0
    for e in edges:
        R += aggregate_local_core_values(list(e), order, local_core_values)
    return R


def get_adj(d):
    adj = {}
    for e in d:
        for n in e:
            if n in adj:
                adj[n].append(tuple(sorted(list(e))))
            else:
                adj[n] = [tuple(sorted(list(e)))]

    return adj


def sort_by_degree(d):
    deg = {}
    for e in d:
        for n in e:
            if n not in deg:
                deg[n] = 0
            deg[n] += 1
    deg_list = []
    for k in deg:
        deg_list.append((deg[k], k))

    deg_list = list(sorted(deg_list))
    return deg_list


def core_periphery(
    hypergraph: Hypergraph,
    greedy_start=False,
    N_ITER=1000,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Implementation of the core-periphery model described in: https://arxiv.org/pdf/2202.12769.pdf

    Parameters
    ----------
    hypergraph: Hypergraph
        Hypergraph object
    greedy_start: bool
        If True, use a greedy approach to find the initial permutation of nodes (default: False)
    N_ITER: int
        Number of iterations (default: 1000)

    Returns
    -------
    dict
        Dictionary with coreness scores for each node
    """
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    pyrand = random.Random(None if seed is None else int(seed))

    if hypergraph.is_weighted():
        w = {}
        for edge in hypergraph.get_edges():
            w[tuple(sorted(edge))] = hypergraph.get_weight(tuple(sorted(edge)))
    else:
        w = {}
        for edge in hypergraph.get_edges():
            w[tuple(sorted(edge))] = 1

    d = list(w.keys())

    N_nodes, node2id, d, w = relabel_nodes(d, w)
    deg_list = sort_by_degree(d)

    adj = get_adj(d)

    cs = {i: 0 for i in range(N_nodes)}

    NUM_SWITCH = N_nodes * 10

    for n_iter in range(N_ITER):
        a, b = sample_params(rng=rng)
        local_core_values = {}

        for i in range(1, N_nodes + 1):
            local_core_values[i - 1] = transition_function(i, N_nodes, a, b)

        order = {}

        # initial permutation
        if greedy_start:
            for i, k in enumerate(deg_list):
                order[k[1]] = i
        else:
            tmp = [i for i in range(N_nodes)]
            pyrand.shuffle(tmp)
            for i in range(N_nodes):
                order[i] = tmp[i]

        R = get_core_quality(d, order, local_core_values)

        # label switching
        for _ in range(NUM_SWITCH):
            i, j = pyrand.sample(range(N_nodes), 2)

            new_R = R

            for e in adj[i]:
                new_R -= (
                    w[e]
                    / len(e)
                    * aggregate_local_core_values(list(e), order, local_core_values)
                )

            for e in adj[j]:
                new_R -= (
                    w[e]
                    / len(e)
                    * aggregate_local_core_values(list(e), order, local_core_values)
                )

            s_tmp = order[i]
            order[i] = order[j]
            order[j] = s_tmp

            for e in adj[i]:
                new_R += (
                    w[e]
                    / len(e)
                    * aggregate_local_core_values(list(e), order, local_core_values)
                )

            for e in adj[j]:
                new_R += (
                    w[e]
                    / len(e)
                    * aggregate_local_core_values(list(e), order, local_core_values)
                )

            if new_R < R:
                s_tmp = order[i]
                order[i] = order[j]
                order[j] = s_tmp
            else:
                R = new_R

        for node in range(N_nodes):
            cs[node] = cs[node] + local_core_values[order[node]] * R

        logger = logging.getLogger(__name__)
        logger.info("%s of %s iter", n_iter, N_ITER)

    max_node = max(cs, key=cs.get)
    Z = 1 / cs[max_node]

    for node in range(N_nodes):
        cs[node] = Z * cs[node]

    id2node = {}
    for node in node2id:
        id2node[node2id[node]] = node

    return {id2node[i]: cs[i] for i in cs}
