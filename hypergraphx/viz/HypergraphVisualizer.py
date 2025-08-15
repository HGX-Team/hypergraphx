import random
import networkx as nx
import numpy as np

from hypergraphx.core.Hypergraph import Hypergraph
from hypergraphx.viz.IHypergraphVisualizer import IHypergraphVisualizer
from hypergraphx.viz.Object import Object
from hypergraphx.linalg import *
from hypergraphx.generation.random import *

class HypergraphVisualizer(IHypergraphVisualizer):
    def __init__(self, g: Hypergraph):
        super().__init__(g=g)
        self.directed = False
    
    def to_nx(self) -> nx.Graph:
        return self.get_pairwise_subgraph()

    def get_hyperedge_labels(self, key:str="label") -> Dict[tuple, str]:
        """
        Get hyperedge labels for visualization.
        """
        return {
            edge: metadata.get(key, '')
            for edge, metadata in self.g.get_edges(metadata=True).items()
            if key in metadata.keys() and len(edge) > 2
        }

    def get_hyperedge_styling_data(
            self,
            hye: Tuple[int],
            pos: Dict[int, tuple],
            number_of_refinements: int = 12
        ) -> Tuple[Tuple[float, float], Tuple[List[float], List[float]]]:
        """
        Get the fill data for a hyperedge.
        """
        # Center of mass of points.
        points, x_c, y_c = self.get_hyperedge_center_of_mass(pos, hye)

        # Order points in a clockwise fashion.
        points = sorted(
            points,
            key=lambda point: np.arctan2(point[1] - y_c, point[0] - x_c)
        )

        offset_multiplier = 2.5 if len(points) == 3 else 1.8
        points = [
            (
                x_c + offset_multiplier * (x - x_c),
                y_c + offset_multiplier * (y - y_c)
            ) for x, y in points
        ]
        # append the starting point to the cartesian coords list so it corresponds to a polygon
        if points[0] != points[-1]:
            points.append(points[0])
        obj = Object(points)
        smoothed_obj_coords = obj.Smooth_by_Chaikin(number_of_refinements)
        
        order = len(hye) - 1
        if order not in self.hyperedge_color_by_order.keys():
            std_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
            self.hyperedge_color_by_order[order] = std_color

        if order not in self.hyperedge_facecolor_by_order.keys():
            std_face_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
            self.hyperedge_facecolor_by_order[order] = std_face_color

        # Extract x and y coordinates from the smoothed object.
        return (
            (x_c, y_c),
            (
                [pt[0] for pt in smoothed_obj_coords],
                [pt[1] for pt in smoothed_obj_coords]
            )
        )