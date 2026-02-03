from hypergraphx.core.temporal import TemporalHypergraph
import numpy as np

from hypergraphx.generation._rng import np_rng, py_rng


def rnd_pwl(xmin, xmax, g, size=1, *, seed: int | None = None):
    rng = np_rng(seed)
    r = rng.random(size=size)
    return (r * (xmax ** (1.0 - g) - xmin ** (1.0 - g)) + xmin ** (1.0 - g)) ** (
        1.0 / (1.0 - g)
    )


def HOADmodel(N: int, activities_per_order: dict, time=100, *, seed: int | None = None):
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

    Examples
    --------
    >>> import numpy as np
    >>> from hypergraphx.generation import HOADmodel
    >>> acts = {2: np.array([0.1, 0.2, 0.3])}  # order=2 => size=3 hyperedges
    >>> T = HOADmodel(3, acts, time=5, seed=0)
    >>> isinstance(T.get_edges(), list)
    True
    """
    rng = py_rng(seed)

    hyperlinks = []
    for order in activities_per_order.keys():
        act_vect = activities_per_order[order]
        for t in range(time):
            for node_i in range(N):
                if act_vect[node_i] > rng.random():
                    neigh_list = rng.sample(range(N), order)
                    neigh_list.append(node_i)
                    if len(neigh_list) == len(set(neigh_list)):
                        hyperlinks.append((t, tuple(neigh_list)))
    HG = TemporalHypergraph(hyperlinks)
    return HG
