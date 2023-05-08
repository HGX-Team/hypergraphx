from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.representations.projections import clique_projection


def draw_communities(
    hypergraph: Hypergraph,
    u: np.array,
    col: dict,
    figsize: tuple = (7, 7),
    ax: Optional[plt.Axes] = None,
    pos: Optional[dict] = None,
    edge_color: str = "lightgrey",
    edge_width: float = 0.3,
    threshold_group: float = 0.1,
    wedge_color: str = "lightgrey",
    wedge_width: float = 1.5,
    with_node_labels: bool = True,
    label_size: float = 10,
    label_col: str = "black",
    node_size: Union[None, float, int, dict] = None,
    c_node_size: float = 0.004,
    title: Optional[str] = None,
    title_size: float = 15,
    seed: int = 20,
    scale: int = 2,
    iterations: int = 100,
    opt_dist: float = 0.2,
):
    """Visualize the node memberships of a hypergraph. Nodes are colored according to their memberships,
    which can be either hard- or soft-membership, and the node size is proportional to the degree in the hypergraph.
    Edges are the pairwise interactions of the hypergraph clique projection.

    Parameters
    ----------
    hypergraph: the hypergraph to visualize.
    u: membership matrix of dimension NxK, where N is the number of nodes and K is the number of communities.
    col: dictionary of colors for nodes, where key represent the group id and values are colors.
    figsize: size of the figure to use when ax=None.
    ax: axes to use for the visualization.
    pos: dictionary of positions for nodes, with node as keys and values as a coordinate list or tuple.
    edge_color: color of the edges.
    edge_width: width of the edges.
    threshold_group: minimum membership value to keep in the plot.
    wedge_color: color of the wedge borders.
    wedge_width: width of the wedge borders.
    with_node_labels: flag to print the node labels.
    label_size: fontsize of the node labels.
    label_col: color of the node labels.
    node_size: sizes of nodes.
    c_node_size: constant to regularize the node size proportional to the node degree (when node_size=None).
    title: plot title.
    title_size: fontsize of the title.
    seed: random seed for fixing the position with the spring layout.
    scale: scale factor for positions.
    iterations: maximum number of iterations taken for fixing the position with the spring layout.
    opt_dist: optimal distance between nodes.
    """
    # Initialize figure.
    if ax is None:
        plt.figure(figsize=figsize)
        plt.subplot(1, 1, 1)
        ax = plt.gca()

    # Get the clique projection of the hypergraph.
    G = clique_projection(hypergraph, keep_isolated=True)

    # Extract node positions.
    if pos is None:
        pos = nx.spring_layout(
            G, k=opt_dist, iterations=iterations, seed=seed, scale=scale
        )

    # Get node degrees and node sizes.
    degree = hypergraph.degree_sequence()
    if node_size is None:
        # Proportional to the node degree.
        node_size = {n: degree[n] * c_node_size for n in G.nodes()}
    elif type(node_size) != np.array:
        node_size = {n: node_size for n in G.nodes()}

    # Get node mappings.
    _, mappingID2Name = hypergraph.binary_incidence_matrix(return_mapping=True)
    mappingName2ID = {n: i for i, n in mappingID2Name.items()}

    # Plot edges.
    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color=edge_color, ax=ax)

    # Plot nodes.
    for n in G.nodes():
        wedge_sizes, wedge_colors = extract_pie_properties(
            mappingName2ID[n], u, col, threshold=threshold_group
        )
        if len(wedge_sizes) > 0:
            plt.pie(
                wedge_sizes,
                center=pos[n],
                colors=wedge_colors,
                radius=node_size[n],
                wedgeprops={"edgecolor": wedge_color, "linewidth": wedge_width},
                normalize=True,
            )
            if with_node_labels:
                ax.annotate(
                    n,
                    (pos[n][0] - 0.1, pos[n][1] - 0.06),
                    fontsize=label_size,
                    color=label_col,
                )
            if title is not None:
                plt.title(title, fontsize=title_size)
            ax.axis("equal")
            plt.axis("off")
    plt.tight_layout()


def extract_pie_properties(
    i: int, u: np.array, colors: dict, threshold: float = 0.1
) -> Tuple[np.array, np.array]:
    """Given a node, it extracts the wedge sizes and the respective colors for the pie chart
    that represents its membership.

    Parameters
    ----------
    i: node id.
    u: membership matrix.
    colors: dictionary of colors, where key represent the group id and values are colors.
    threshold: threshold for node membership.

    Returns
    -------
    wedge_sizes: wedge sizes.
    wedge_colors: sequence of colors through which the pie chart will cycle.
    """
    valid_groups = np.where(u[i] > threshold)[0]
    wedge_sizes = u[i][valid_groups]
    wedge_colors = [colors[k] for k in valid_groups]
    return wedge_sizes, wedge_colors
