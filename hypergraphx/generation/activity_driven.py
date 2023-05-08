import random

from hypergraphx.core.temporal_hypergraph import TemporalHypergraph
from hypergraphx.dynamics.randwalk import *


def rnd_pwl(xmin, xmax, g, size=1):
    r = np.random.random(size=size)
    return (r * (xmax ** (1. - g) - xmin ** (1. - g)) + xmin ** (1. - g)) ** (1. / (1. - g))


def HOADmodel(N: int, activities_per_order: dict, time = 100):
    """
    Generate a temporal hypergraph according to the HOAD model.

    Parameters
    ----------
    N : int
        The number of nodes in the hypergraph.
    activities_per_order : dict
        The dictionary of activities per order. The keys are the orders and the values are the activities.
    time : int
        The number of time steps.

    Returns
    -------
    HG : TemporalHypergraph
        The temporal hypergraph generated according to the HOAD model.

    """

    hyperlinks = []
    for order in activities_per_order.keys():
        act_vect = activities_per_order[order]
        for t in range(time):
            for node_i in range(N):
                if act_vect[node_i] > random.random():
                    neigh_list = random.sample(range(N), order)
                    neigh_list.append(node_i)
                    if len(neigh_list) == len(set(neigh_list)):
                        hyperlinks.append((t, tuple(neigh_list)))
    HG = TemporalHypergraph(hyperlinks)
    return HG