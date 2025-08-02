import random
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hypergraphx.core.hypergraph import Hypergraph
from hypergraphx.viz.IHypergraphVisualizer import IHypergraphVisualizer
from hypergraphx.viz.Object import Object
from hypergraphx.linalg import *
from hypergraphx.generation.random import *

class HypergraphVisualizer(IHypergraphVisualizer):
    def __init__(self, g: Hypergraph):
        super().__init__()
        self.directed = False
    
    def to_nx(self) -> nx.Graph:
        return self.get_pairwise_subgraph()

    def get_hyperedge_labels(self, key:str="type") -> Dict[tuple, str]:
        """
        Get hyperedge labels for visualization.
        """
        return {
            edge: metadata.get(key, '')
            for edge, metadata in self.g.edges(metadata=True).items()
            if key in metadata.keys() and len(edge) > 2
        }

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
        # Center of mass of points.
        x_c, y_c = self.get_hyperedge_center_of_mass(pos, hye)

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