import math

import numpy as np


def count_nodes(d):
    N = set()

    node2id = {}
    idx = 0

    new_d = set()

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

    return len(N), node2id, new_d


def sample_params():
    return np.random.uniform(0, 1), np.random.uniform(0, 1)


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
