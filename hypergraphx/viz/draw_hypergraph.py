import random
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hypergraphx.core.I_hypergraph import IHypergraph
from hypergraphx.viz.Object import Object
from hypergraphx.linalg import *
from hypergraphx.generation.random import *
from hypergraphx.representations.projections import clique_projection


def to_nx(g:IHypergraph) -> nx.DiGraph:
    """
    Convert a directed Hypergraph to a NetworkX Graph.
    """
    G = nx.DiGraph()
    for node in g.get_nodes():
        G.add_node(
            node,
            **g.get_node_metadata(node)
        )
    
    for edge in g.get_edges(order=1, metadata=True):
        G.add_edge(
            edge,
            **g.get_edge_metadata(edge)
        )
    
    return G

def get_node_labels(
        g: IHypergraph,
        key:str="text") -> Dict[int, str]:
    """
    Get node labels for visualization.
    """
    return {
        node: metadata.get(key, '')
        for node, metadata in g.nodes(
            metadata=True
        ).items()
        if key in metadata.keys()
    }


def get_pairwise_edge_labels(
        g: IHypergraph,
        key:str="type") -> Dict[tuple, str]:
    """
    Get edge labels for edges of order 1 (standard edge pairs) to use for visualization.
    """
    return {
        edge: metadata.get(key, '')
        for edge, metadata in g.get_edges(
            order=1,
            metadata=True
        ).items()
        if key in metadata.keys()
    }


def get_hyperedge_labels(
        g: IHypergraph,
        key:str="type") -> Dict[tuple, str]:
    """
    Get hyperedge labels for visualization.
    """
    return {
        edge: metadata.get(key, '')
        for edge, metadata in g.edges(metadata=True).items()
        if key in metadata.keys() and len(edge) > 2
    }


def draw(
    # main parameters
    hypergraph: IHypergraph,
    figsize: tuple = (12, 7),
    ax: Optional[plt.Axes] = None,

    # node position parameters
    pos: Optional[dict] = None,
    iterations: int = 100,
    seed: int = 10,
    scale: int = 1,
    opt_dist: float = 0.5,

    # node styling
    with_node_labels: bool = False,
    node_size: Union[int, np.array] = 150,
    node_color: Union[str, np.array] = "#E2E0DD",
    node_facecolor: Union[str, np.array] = "black",
    node_shape: str = "o",

    # edge styling
    with_pairwise_edge_labels: bool = False,
    pairwise_edge_color: str = "lightgrey",
    pairwise_edge_width: float = 1.2,
    
    # hyperedge styling
    # Set color hyperedges of size > 2 (order > 1).
    with_hyperedge_labels: bool = False,
    hyperedge_color_by_order: dict = {2: "#FFBC79", 3: "#79BCFF", 4: "#4C9F4C"},
    hyperedge_facecolor_by_order: dict = {2: "#FFBC79", 3: "#79BCFF", 4: "#4C9F4C"},
    hyperedge_alpha: Union[float, np.array] = 0.8,

    # other styling parameters
    label_size: float = 10,
    label_col: str = "black",
):
    """Visualize a hypergraph."""
    # Initialize figure.
    if ax is None:
        plt.figure(figsize=figsize)
        plt.subplot(1, 1, 1)
        ax = plt.gca()

    # Extract node positions based on the hypergraph clique projection.
    if pos is None:
        pos = nx.spring_layout(
            G=clique_projection(
                hypergraph,
                keep_isolated=True
            ), # type: ignore
            iterations=iterations,
            seed=seed,
            scale=scale,
            k=opt_dist,
        )


    # Initialize a networkx graph with the nodes and only the pairwise interactions of the hypergraph.
    pairwise_G = to_nx(hypergraph)

    # Plot the pairwise graph.
    if type(node_shape) == str:
        node_shape = {n: node_shape for n in pairwise_G.nodes()}
    for nid, n in enumerate(list(pairwise_G.nodes())):
        nx.draw_networkx_nodes(
            G=pairwise_G,
            pos=pos,
            nodelist=[n],
            node_size=node_size,
            node_shape=node_shape[n],
            node_color=node_color,
            alpha=0.8,
            linewidths=2,
            edgecolors=node_facecolor,
            ax=ax,
        )
    if with_node_labels:
        nx.draw_networkx_labels(
            G=pairwise_G,
            pos=pos,
            labels=get_node_labels(hypergraph),
            font_size=int(label_size),
            font_color=label_col,
            ax=ax,
        )

    # Plot the edges of the pairwise graph.
    if with_pairwise_edge_labels:
        nx.draw_networkx_edge_labels(
            G=pairwise_G,
            pos=pos,
            edge_labels=get_pairwise_edge_labels(hypergraph),
            font_size=int(label_size),
            font_color=label_col,
            ax=ax,
        )
    nx.draw_networkx_edges(
        G=pairwise_G,
        pos=pos,
        width=pairwise_edge_width,
        edge_color=pairwise_edge_color,
        alpha=0.8,
        ax=ax,
    )

    # Plot the hyperedges (size>2/order>1).
    hyperedge_labels = get_hyperedge_labels(hypergraph) if with_hyperedge_labels else dict()
    for hye in list(hypergraph.get_edges()):
        if len(hye) > 2:
            x1, y1, color, facecolor = get_hyperedge_styling_data(
                hye,
                pos,
                hyperedge_color_by_order,
                hyperedge_facecolor_by_order
            )
            ax.fill(
                x1,
                y1,
                alpha=hyperedge_alpha,
                c=color,
                edgecolor=facecolor
            )
            if with_hyperedge_labels:
                ax.annotate(
                    hyperedge_labels.get(hye, ''),
                    (
                        pos[n][0] - 0.1,
                        pos[n][1] - 0.06
                    ),
                    fontsize=label_size,
                    color=label_col,
                )
    
    # Set the aspect ratio of the plot to be equal.
    ax.axis("equal")
    plt.axis("equal")
    plt.title("Semantic Knowledge Graph")
    plt.show()


def get_hyperedge_styling_data(
        hye,
        pos: Dict[int, tuple],
        hyperedge_color_by_order: Dict[int, str],
        hyperedge_facecolor_by_order: Dict[int, str]
    ) -> tuple[List[float], List[float], str, str]:
    """
    Get the fill data for a hyperedge.
    """
        
    # Center of mass of points.
    points = [
        (
            pos[node][0],
            pos[node][1]
        ) for node in hye
    ]
    x_c = np.mean([x for x, y in points])
    y_c = np.mean([y for x, y in points])

    # Order points in a clockwise fashion.
    points = sorted(
        points,
        key=lambda x: np.arctan2(x[1] - y_c, x[0] - x_c)
    )

    if len(points) == 3:
        points = [
            (
                x_c + 2.5 * (x - x_c),
                y_c + 2.5 * (y - y_c)
            ) for x, y in points
        ]
    else:
        points = [
            (
                x_c + 1.8 * (x - x_c),
                y_c + 1.8 * (y - y_c)
            ) for x, y in points
        ]
    Cartesian_coords_list = points + [points[0]]

    obj = Object(Cartesian_coords_list)
    Smoothed_obj = obj.Smooth_by_Chaikin(number_of_refinements=12)
    
    # Extract x and y coordinates from the smoothed object.
    x1 = [i for i, j in Smoothed_obj]
    y1 = [j for i, j in Smoothed_obj]

    order = len(hye) - 1

    if order not in hyperedge_color_by_order.keys():
        std_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        hyperedge_color_by_order[order] = std_color

    if order not in hyperedge_facecolor_by_order.keys():
        std_face_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        hyperedge_facecolor_by_order[order] = std_face_color

    color = hyperedge_color_by_order[order]
    facecolor = hyperedge_facecolor_by_order[order]

    return x1, y1, color, facecolor