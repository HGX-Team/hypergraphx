import random
from typing import Optional, Union

import networkx as nx
import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.representations.projections import clique_projection


def sum_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return x1 + x2, y1 + y2


def multiply_point(multiplier, point):
    x, y = point
    return float(x) * float(multiplier), float(y) * float(multiplier)


def is_polygon(cartesian_coords_list):
    if (
        cartesian_coords_list[0]
        == cartesian_coords_list[len(cartesian_coords_list) - 1]
    ):
        return True
    else:
        return False


class Object:
    def __init__(self, cartesian_coords_list):
        self.Cartesian_coords_list = cartesian_coords_list

    def Find_Q_point_position(self, P1, P2):
        Summand1 = multiply_point(float(3) / float(4), P1)
        Summand2 = multiply_point(float(1) / float(4), P2)
        Q = sum_points(Summand1, Summand2)
        return Q

    def Find_R_point_position(self, P1, P2):
        Summand1 = multiply_point(float(1) / float(4), P1)
        Summand2 = multiply_point(float(3) / float(4), P2)
        R = sum_points(Summand1, Summand2)
        return R

    def Smooth_by_Chaikin(self, number_of_refinements):
        refinement = 1
        copy_first_coord = is_polygon(self.Cartesian_coords_list)
        obj = Object(self.Cartesian_coords_list)
        while refinement <= number_of_refinements:
            self.New_cartesian_coords_list = []

            for num, tuple in enumerate(self.Cartesian_coords_list):
                if num + 1 == len(self.Cartesian_coords_list):
                    pass
                else:
                    P1, P2 = (tuple, self.Cartesian_coords_list[num + 1])
                    Q = obj.Find_Q_point_position(P1, P2)
                    R = obj.Find_R_point_position(P1, P2)
                    self.New_cartesian_coords_list.append(Q)
                    self.New_cartesian_coords_list.append(R)

            if copy_first_coord:
                self.New_cartesian_coords_list.append(self.New_cartesian_coords_list[0])

            self.Cartesian_coords_list = self.New_cartesian_coords_list
            refinement += 1
        return self.Cartesian_coords_list


def draw_hypergraph(
    hypergraph: Hypergraph,
    figsize: tuple = (12, 7),
    ax: Optional["plt.Axes"] = None,
    pos: Optional[dict] = None,
    edge_color: str = "lightgrey",
    hyperedge_color_by_order: Optional[dict] = None,
    hyperedge_facecolor_by_order: Optional[dict] = None,
    edge_width: float = 1.2,
    hyperedge_alpha: Union[float, np.array] = 0.8,
    node_size: Union[int, np.array] = 150,
    node_color: Union[str, np.array] = "#E2E0DD",
    node_facecolor: Union[str, np.array] = "black",
    node_shape: str = "o",
    with_node_labels: bool = False,
    label_size: float = 10,
    label_col: str = "black",
    seed: int = 10,
    scale: int = 1,
    iterations: int = 100,
    opt_dist: float = 0.5,
    show: bool = False,
):
    """Visualize a hypergraph.

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
            "draw_hypergraph requires matplotlib. Install with `pip install hypergraphx[viz]`."
        ) from exc

    def _stable_color(order_value):
        rng = random.Random(order_value)
        return f"#{rng.randrange(0x1000000):06x}"

    # Initialize figure.
    if ax is None:
        plt.figure(figsize=figsize)
        plt.subplot(1, 1, 1)
        ax = plt.gca()

    # Extract node positions based on the hypergraph clique projection.
    if pos is None:
        pos = nx.spring_layout(
            clique_projection(hypergraph, keep_isolated=True),
            iterations=iterations,
            seed=seed,
            scale=scale,
            k=opt_dist,
        )
    else:
        missing_nodes = set(hypergraph.get_nodes()) - set(pos.keys())
        if missing_nodes:
            raise ValueError("pos is missing positions for some nodes.")

    # Set color hyperedges of size > 2 (order > 1).
    if hyperedge_color_by_order is None:
        hyperedge_color_by_order = {2: "#FFBC79", 3: "#79BCFF", 4: "#4C9F4C"}
    else:
        hyperedge_color_by_order = dict(hyperedge_color_by_order)
    if hyperedge_facecolor_by_order is None:
        hyperedge_facecolor_by_order = {2: "#FFBC79", 3: "#79BCFF", 4: "#4C9F4C"}
    else:
        hyperedge_facecolor_by_order = dict(hyperedge_facecolor_by_order)

    # Extract edges (hyperedges of size=2/order=1).
    edges = hypergraph.get_edges(order=1)

    # Initialize empty graph with the nodes and the pairwise interactions of the hypergraph.
    G = nx.Graph()
    G.add_nodes_from(hypergraph.get_nodes())
    for e in edges:
        G.add_edge(e[0], e[1])

    nodes = list(G.nodes())
    if isinstance(node_shape, str):
        node_shape = {n: node_shape for n in nodes}
    if isinstance(node_shape, dict):
        missing_shapes = set(nodes) - set(node_shape.keys())
        if missing_shapes:
            raise ValueError("node_shape is missing entries for some nodes.")
    if isinstance(node_size, np.ndarray):
        if len(node_size) != len(nodes):
            raise ValueError("node_size length must match number of nodes.")
        node_size = {n: node_size[i] for i, n in enumerate(nodes)}
    elif not isinstance(node_size, dict):
        node_size = {n: node_size for n in nodes}
    if isinstance(node_color, np.ndarray):
        if len(node_color) != len(nodes):
            raise ValueError("node_color length must match number of nodes.")
        node_color = {n: node_color[i] for i, n in enumerate(nodes)}
    elif not isinstance(node_color, dict):
        node_color = {n: node_color for n in nodes}
    if isinstance(node_facecolor, np.ndarray):
        if len(node_facecolor) != len(nodes):
            raise ValueError("node_facecolor length must match number of nodes.")
        node_facecolor = {n: node_facecolor[i] for i, n in enumerate(nodes)}
    elif not isinstance(node_facecolor, dict):
        node_facecolor = {n: node_facecolor for n in nodes}

    for n in nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            [n],
            node_size=node_size[n],
            node_shape=node_shape[n],
            node_color=node_color[n],
            edgecolors=node_facecolor[n],
            ax=ax,
        )
        if with_node_labels:
            ax.annotate(
                n,
                (pos[n][0] - 0.1, pos[n][1] - 0.06),
                fontsize=label_size,
                color=label_col,
            )

    # Plot the hyperedges (size>2/order>1).
    for hye in list(hypergraph.get_edges()):
        if len(hye) > 2:
            points = []
            for node in hye:
                points.append((pos[node][0], pos[node][1]))
                # Center of mass of points.
                x_c = np.mean([x for x, y in points])
                y_c = np.mean([y for x, y in points])
            # Order points in a clockwise fashion.
            points = sorted(points, key=lambda x: np.arctan2(x[1] - y_c, x[0] - x_c))

            if len(points) == 3:
                points = [
                    (x_c + 2.5 * (x - x_c), y_c + 2.5 * (y - y_c)) for x, y in points
                ]
            else:
                points = [
                    (x_c + 1.8 * (x - x_c), y_c + 1.8 * (y - y_c)) for x, y in points
                ]
            Cartesian_coords_list = points + [points[0]]

            obj = Object(Cartesian_coords_list)
            Smoothed_obj = obj.Smooth_by_Chaikin(number_of_refinements=12)

            # Visualisation.
            x1 = [i for i, j in Smoothed_obj]
            y1 = [j for i, j in Smoothed_obj]

            order = len(hye) - 1

            if order not in hyperedge_color_by_order.keys():
                hyperedge_color_by_order[order] = _stable_color(order)

            if order not in hyperedge_facecolor_by_order.keys():
                hyperedge_facecolor_by_order[order] = _stable_color(order + 1000)

            color = hyperedge_color_by_order[order]
            facecolor = hyperedge_facecolor_by_order[order]
            ax.fill(x1, y1, alpha=hyperedge_alpha, c=color, edgecolor=facecolor)

    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color=edge_color, ax=ax)

    ax.axis("equal")
    plt.axis("equal")
    if show:
        plt.show()
    return ax
