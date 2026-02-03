import networkx as nx

from hypergraphx import Hypergraph
from hypergraphx.representations.projections import (
    bipartite_projection,
    clique_projection,
)


def draw_bipartite(
    h: Hypergraph, pos=None, ax=None, align="vertical", show: bool = False, **kwargs
):
    """
    Draws a bipartite graph representation of the hypergraph.
    Parameters
    ----------
    h : Hypergraph.
        The hypergraph to be projected.
    pos : dict.
        A dictionary with nodes as keys and positions as values.
    ax : matplotlib.axes.Axes.
        The axes to draw the graph on.
    kwargs : dict.
        Keyword arguments to be passed to networkx.draw_networkx.
    align : str.
        The alignment of the nodes. Can be 'vertical' or 'horizontal'.
    show : bool.
        If True, call plt.show().

    Returns
    -------
    ax : matplotlib.axes.Axes.
        The axes the graph was drawn on.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "draw_bipartite requires matplotlib. Install with `pip install hypergraphx[viz]`."
        ) from exc

    g, _ = bipartite_projection(h)

    if pos is None:
        pos = nx.bipartite_layout(
            g, nodes=[n for n, d in g.nodes(data=True) if d["bipartite"] == 0]
        )
    else:
        missing_nodes = set(g.nodes()) - set(pos.keys())
        if missing_nodes:
            raise ValueError("pos is missing positions for some nodes.")
    if align not in {"vertical", "horizontal"}:
        raise ValueError("align must be 'vertical' or 'horizontal'.")
    if align == "horizontal":
        pos = {n: (y, x) for n, (x, y) in pos.items()}

    if ax is None:
        ax = plt.gca()

    nx.draw_networkx(g, pos=pos, ax=ax, **kwargs)
    if show:
        plt.show()
    return ax


def draw_clique(h: Hypergraph, pos=None, ax=None, show: bool = False, **kwargs):
    """
    Draws a clique projection of the hypergraph.
    Parameters
    ----------
    h : Hypergraph.
        The hypergraph to be projected.
    pos : dict.
        A dictionary with nodes as keys and positions as values.
    ax : matplotlib.axes.Axes.
        The axes to draw the graph on.
    kwargs : dict.
        Keyword arguments to be passed to networkx.draw_networkx.
    show : bool.
        If True, call plt.show().

    Returns
    -------
    ax : matplotlib.axes.Axes. The axes the graph was drawn on.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "draw_clique requires matplotlib. Install with `pip install hypergraphx[viz]`."
        ) from exc

    g = clique_projection(h)

    if pos is None:
        pos = nx.spring_layout(g)
    else:
        missing_nodes = set(g.nodes()) - set(pos.keys())
        if missing_nodes:
            raise ValueError("pos is missing positions for some nodes.")

    if ax is None:
        ax = plt.gca()

    nx.draw_networkx(g, pos=pos, ax=ax, **kwargs)
    if show:
        plt.show()
    return ax
