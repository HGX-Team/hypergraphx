from abc import ABC, abstractmethod
import random
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hypergraphx.core.IHypergraph import IHypergraph
from hypergraphx.viz.Object import Object
from hypergraphx.linalg import *
from hypergraphx.generation.random import *
from hypergraphx.representations.projections import clique_projection

class IHypergraphVisualizer(ABC):
    def __init__(self, g: IHypergraph):
        self.g = g
        self.directed = None
        self.node_labels = self.get_node_labels()
        self.pairwise_edge_labels = self.get_pairwise_edge_labels()
        self.hyperedge_labels = self.get_hyperedge_labels()

    @abstractmethod
    def to_nx(self, pairwise_only: bool=True) -> nx.DiGraph | nx.Graph:
        pass

    # TODO: Fix this to avoid wrapping node id's during edge creation:
    #  want (a, b)
    #  currently ((a,),(b,))?
    def get_pairwise_subgraph(self) -> nx.DiGraph | nx.Graph:
        """
        Convert a directed/undirected Hypergraph to a NetworkX Graph.
        """
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        for node in self.g.get_nodes():
            G.add_node(
                node,
                **self.g.get_node_metadata(node)
            )
        
        for edge in self.g.get_edges(order=1, metadata=True):
            G.add_edge(
                edge[0],
                edge[1],
                **self.g.get_edge_metadata(edge)
            )
        
        return G

    def get_node_labels(self, key:str="text") -> Dict[int, str]:
        """
        Get node labels for visualization.
        """
        return {
            node: metadata.get(key, '')
            for node, metadata in self.g.nodes(
                metadata=True
            ).items()
            if key in metadata.keys()
        }


    def get_pairwise_edge_labels(self, key:str="type") -> Dict[tuple, str]:
        """
        Get edge labels for edges of order 1 (standard edge pairs) to use for visualization.
        """
        return {
            edge: metadata.get(key, '')
            for edge, metadata in self.g.get_edges(
                order=1,
                metadata=True
            ).items()
            if key in metadata.keys()
        }

    @abstractmethod
    def get_hyperedge_labels(self, key:str="type") -> Dict[tuple, str]:
        """
        Get hyperedge labels for visualization.
        """
        pass

    def get_hyperedge_center_of_mass(self,
            pos,
            hye
        ) -> Tuple[int, int]:
        points = [
            (
                pos[node][0],
                pos[node][1]
            ) for node in hye
        ]
        x_c = np.mean([x for x, y in points])
        y_c = np.mean([y for x, y in points])
        return (x_c, y_c)

    @abstractmethod
    def get_hyperedge_styling_data(
            self,
            hye,
            pos: Dict[int, tuple],
            hyperedge_color_by_order: Dict[int, str],
            hyperedge_facecolor_by_order: Dict[int, str]
        ) -> tuple[List[float], List[float], str, str]:
        """
        Get the fill data for a hyperedge.
        """
        pass


    def draw_pairwise_G(self,
        # main parameters
        ax: Optional[plt.Axes] = None,
        pos: Optional[dict] = None,

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
        
        # other styling parameters
        label_size: float = 10,
        label_col: str = "black",
    ):
        
        # Initialize a networkx graph with the nodes and only the pairwise interactions of the hypergraph.
        pairwise_G = self.to_nx(pairwise_only=True)
        
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
        if self.node_labels:
            nx.draw_networkx_labels(
                G=pairwise_G,
                pos=pos,
                labels=self.node_labels,
                font_size=int(label_size),
                font_color=label_col,
                ax=ax,
            )

        # Plot the edges of the pairwise graph.
        if self.pairwise_edge_labels:
            nx.draw_networkx_edge_labels(
                G=pairwise_G,
                pos=pos,
                edge_labels=self.pairwise_edge_labels,
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

    def draw_hyperedges(self,
        # main parameters
        ax: Optional[plt.Axes] = None,
        pos: Optional[dict] = None,

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
        for hye in list(self.g.get_edges()):
            if len(hye) > 2:
                x1, y1, color, facecolor = self.get_hyperedge_styling_data(
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
                        self.hyperedge_labels.get(hye, ''),
                        (x1, y1),
                        fontsize=label_size,
                        color=label_col,
                    )

    def draw(self,
        # main parameters
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
        pairwise_edge_labels: List[str] = None,
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

        # other parameters
        plot_title: str = "Hypergraph"
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
                    self.g,
                    keep_isolated=True
                ), # type: ignore
                iterations=iterations,
                seed=seed,
                scale=scale,
                k=opt_dist,
            )


        # Plot the subgraph of nodes and their pairwise edges ONLY
        self.draw_pairwise_G(
            # main parameters
            ax=ax,
            pos=pos,

            # node styling
            with_node_labels=with_node_labels,
            node_size=node_size,
            node_color=node_color,
            node_facecolor=node_facecolor,
            node_shape=node_shape,

            # edge styling
            with_pairwise_edge_labels=with_pairwise_edge_labels,
            pairwise_edge_color=pairwise_edge_color,
            pairwise_edge_width=pairwise_edge_width,
            
            # other styling parameters
            label_size=label_size,
            label_col=label_col,
        )
        

        # Plot the hyperedges (size>2/order>1).
        self.draw_hyperedges(
            # main parameters
            ax=ax,
            pos=pos,

            # hyperedge styling
            # Set color hyperedges of size > 2 (order > 1).
            with_hyperedge_labels=with_hyperedge_labels,
            hyperedge_color_by_order=hyperedge_color_by_order,
            hyperedge_facecolor_by_order=hyperedge_facecolor_by_order,
            hyperedge_alpha=hyperedge_alpha,

            # other styling parameters
            label_size=label_size,
            label_col=label_col
        )
        
        # Set the aspect ratio of the plot to be equal.
        ax.axis("equal")
        plt.axis("equal")
        plt.title(plot_title)
        plt.show()