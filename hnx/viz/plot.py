import matplotlib.pyplot as plt
import seaborn as sn

from hnx import Hypergraph
from hnx.measures import degree_distribution


def plot_degree_distribution(hg: Hypergraph, order=None, size=None, ax=None, **kwargs):
    """
    Plots the degree distribution of the hypergraph.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    order : int
        The order of the hyperedges to consider. If None, all hyperedges are considered.
    size : int
        The size of the hyperedges to consider. If None, all hyperedges are considered.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    **kwargs
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was made.
    """
    if ax is None:
        fig, ax = plt.subplots()

    degrees = degree_distribution(hg, order=order, size=size)
    if size is not None:
        ax.set_title("Degree distribution at size {}".format(size))
    if order is not None:
        ax.set_title("Degree distribution at order {}".format(order))

    sn.scatterplot(x=degrees.keys(), y=degrees.values(), ax=ax, **kwargs)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.set_title("Degree distribution")
    return ax


def plot_degree_distributions(hg: Hypergraph, max_size=5, ax=None, **kwargs):
    """
    Plots the degree distributions of the hypergraph at orders 1, 2, 3 and 4.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    **kwargs
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was made.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for size in range(2, max_size+1):
        degrees = degree_distribution(hg, size=size)
        sn.scatterplot(x=degrees.keys(), y=degrees.values(), label="Size: {}".format(size), ax=ax, **kwargs)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    sn.despine()
    return ax

