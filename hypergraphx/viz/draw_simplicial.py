import networkx as nx

from hypergraphx.representations.projections import clique_projection


def find_triplets(list):
    triplets = []
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            for k in range(j + 1, len(list)):
                triplets.append([list[i], list[j], list[k]])
    return triplets


def draw_simplicial(
    HG,
    pos=None,
    link_color="black",
    hyperlink_color_by_order=None,
    link_width=2,
    node_size=150,
    node_color="#5494DA",
    with_labels=False,
    ax=None,
    show: bool = False,
):
    """Draw a simplicial-complex-style visualization for a hypergraph.

    Parameters
    ----------
    show : bool
        If True, call plt.show().

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "draw_simplicial requires matplotlib. Install with `pip install hypergraphx[viz]`."
        ) from exc

    G = clique_projection(HG, keep_isolated=True)
    if pos is None:
        pos = nx.spring_layout(G)
    else:
        missing_nodes = set(G.nodes()) - set(pos.keys())
        if missing_nodes:
            raise ValueError("pos is missing positions for some nodes.")
    if hyperlink_color_by_order is None:
        hyperlink_color_by_order = {2: "r", 3: "orange", 4: "green"}
    else:
        hyperlink_color_by_order = dict(hyperlink_color_by_order)
    for h_edge in HG.get_edges():
        if len(h_edge) > 2:
            order = len(h_edge) - 1

            if order >= 5:
                alpha = 0.1
            else:
                alpha = 0.5

            if order not in hyperlink_color_by_order.keys():
                hyperlink_color_by_order[order] = "Black"
            color = hyperlink_color_by_order[order]

            x_coor = []
            y_coor = []
            triplets = find_triplets(h_edge)
            for triplet in triplets:
                for node in triplet:
                    x_coor.append(pos[node][0])
                    y_coor.append(pos[node][1])
                # print(triplet)

                plt.fill(x_coor, y_coor, alpha=alpha, c=color)

    nx.draw(
        G,
        pos=pos,
        with_labels=with_labels,
        node_color=node_color,
        edge_color=link_color,
        width=link_width,
        node_size=node_size,
        ax=ax,
    )
    if show:
        plt.show()
    return ax


def draw_SC(*args, **kwargs):
    return draw_simplicial(*args, **kwargs)
