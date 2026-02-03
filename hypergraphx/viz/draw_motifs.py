import networkx as nx
import itertools


def draw_motifs(
    patterns,
    edge_size_colors=None,
    node_labels=None,
    node_size=500,
    node_color="lightblue",
    edge_color="black",
    save_path=None,
    axes=None,
    figsize=None,
    tight_layout: bool = True,
    show: bool = False,
):
    """Draw a list of motif patterns side by side.

    Parameters
    ----------
    patterns : list
        List of motif hypergraphs, each expressed as a list of hyperedges.
    axes : matplotlib.axes.Axes or list, optional
        Axes to draw on. If provided, its length must match the number of patterns.
    figsize : tuple, optional
        Figure size used only when axes is None.
    tight_layout : bool
        If True, call plt.tight_layout().
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Polygon  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "draw_motifs requires matplotlib. Install with `pip install hypergraphx[viz]`."
        ) from exc

    # Collect all unique nodes across all patterns
    all_nodes = set(
        itertools.chain.from_iterable(itertools.chain.from_iterable(patterns))
    )
    G_global = nx.Graph()
    G_global.add_nodes_from(all_nodes)
    global_pos = nx.spring_layout(G_global, seed=42)  # consistent layout

    if edge_size_colors is None:
        edge_size_colors = {3: "#FFDAB9", 4: "#ADD8E6"}  # light orange  # light blue

    default_color = "#D3D3D3"  # light gray for other sizes

    edge_sizes = set(len(edge) for graph in patterns for edge in graph if len(edge) > 2)
    for size in edge_sizes:
        if size not in edge_size_colors:
            edge_size_colors[size] = default_color

    num_graphs = len(patterns)
    if axes is None:
        if figsize is None:
            figsize = (5 * num_graphs, 5)
        fig, axes = plt.subplots(1, num_graphs, figsize=figsize)
        if num_graphs == 1:
            axes = [axes]
    else:
        if num_graphs == 1 and not isinstance(axes, (list, tuple)):
            axes = [axes]
        if len(axes) != num_graphs:
            raise ValueError("axes length must match number of patterns.")
        fig = axes[0].figure

    for idx, (hypergraph, ax) in enumerate(zip(patterns, axes)):
        G = nx.Graph()
        nodes = set(itertools.chain.from_iterable(hypergraph))
        G.add_nodes_from(nodes)
        pos = {n: global_pos[n] for n in nodes}

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
            edgecolors="black",
        )

        if node_labels:
            nx.draw_networkx_labels(G, pos, ax=ax)

        # Draw hyperedges
        for i, hedge in enumerate(hypergraph):
            hedge_pos = [pos[n] for n in hedge]
            edge_size = len(hedge)

            if edge_size < 2:
                continue  # skip size-1

            if edge_size == 2:
                # Draw as traditional edge
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[tuple(hedge)],
                    ax=ax,
                    edge_color=edge_color,
                    width=2,
                )
            else:
                color = edge_size_colors[edge_size]
                polygon = Polygon(
                    hedge_pos,
                    closed=True,
                    fill=True,
                    alpha=0.3,
                    facecolor=color,
                    edgecolor=edge_color,
                )
                ax.add_patch(polygon)

        ax.axis("off")

    if tight_layout:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
