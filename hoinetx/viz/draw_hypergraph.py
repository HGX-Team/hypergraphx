import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from hoinetx.representations.projections import *
from hoinetx.core.hypergraph import Hypergraph
def Sum_points(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return x1+x2, y1+y2

def Multiply_point(multiplier, P):
    x, y = P
    return float(x)*float(multiplier), float(y)*float(multiplier)

def Check_if_object_is_polygon(Cartesian_coords_list):
    if Cartesian_coords_list[0] == Cartesian_coords_list[len(Cartesian_coords_list)-1]:
        return True
    else:
        return False

class Object():

    def __init__(self, Cartesian_coords_list):
        self.Cartesian_coords_list = Cartesian_coords_list

    def Find_Q_point_position(self, P1, P2):
        Summand1 = Multiply_point(float(3)/float(4), P1)
        Summand2 = Multiply_point(float(1)/float(4), P2)
        Q = Sum_points(Summand1, Summand2) 
        return Q

    def Find_R_point_position(self, P1, P2):
        Summand1 = Multiply_point(float(1)/float(4), P1)
        Summand2 = Multiply_point(float(3)/float(4), P2)        
        R = Sum_points(Summand1, Summand2)
        return R

    def Smooth_by_Chaikin(self, number_of_refinements):
        refinement = 1
        copy_first_coord = Check_if_object_is_polygon(self.Cartesian_coords_list)
        obj = Object(self.Cartesian_coords_list)
        while refinement <= number_of_refinements:
            self.New_cartesian_coords_list = []

            for num, tuple in enumerate(self.Cartesian_coords_list):
                if num+1 == len(self.Cartesian_coords_list):
                    pass
                else:
                    P1, P2 = (tuple, self.Cartesian_coords_list[num+1])
                    Q = obj.Find_Q_point_position(P1, P2)
                    R = obj.Find_R_point_position(P1, P2)
                    self.New_cartesian_coords_list.append(Q)
                    self.New_cartesian_coords_list.append(R)

            if copy_first_coord:
                self.New_cartesian_coords_list.append(self.New_cartesian_coords_list[0])

            self.Cartesian_coords_list = self.New_cartesian_coords_list
            refinement += 1
        return self.Cartesian_coords_list
    

def draw_HG(HG, pos = None, link_color = 'black',
             hyperlink_color_by_order = {2:'r', 3:'orange', 4:'green'}, 
             link_width = 2, node_size = 150,
             node_color = '#5494DA', alpha = .5,
             with_labels = False,
             ax = None):
    if pos == None:
        pos = nx.spring_layout(clique_projection(HG, keep_isolated=True))
    links = HG.get_edges(order = 1)
    # create a empty graph with nodes = HG.get_nodes()
    G = nx.Graph()
    G.add_nodes_from(HG.get_nodes())
    # add edges to the graph
    for link in links:
        G.add_edge(link[0], link[1])
        
    # loop in HG.get_edges() from the end to the beginning
    
    for h_edge in list(HG.get_edges())[::-1]:
        
        if len(h_edge) > 2:    
            points = []      
            for node in h_edge:
                points.append((pos[node][0], pos[node][1]))
                # center of mass of points
                x_c = np.mean([x for x, y in points])
                y_c = np.mean([y for x, y in points])
                # order points in a clockwise fashion
                
            points = sorted(points, key=lambda x: np.arctan2(x[1] - y_c, x[0] - x_c))

            if len(points) == 3:
                points = [(x_c + 2.5*(x-x_c), y_c + 2.5*(y-y_c)) for x, y in points]
            else:
                points = [(x_c + 1.8*(x-x_c), y_c + 1.8*(y-y_c)) for x, y in points]
            Cartesian_coords_list = points + [points[0]]

            obj = Object(Cartesian_coords_list)    
            Smoothed_obj = obj.Smooth_by_Chaikin(number_of_refinements = 12)

            # visualisation
            x1 = [i for i,j in Smoothed_obj]
            y1 = [j for i,j in Smoothed_obj]

            order = len(h_edge) - 1
            if order not in hyperlink_color_by_order.keys():
                hyperlink_color_by_order[order] = 'Black'
            color = hyperlink_color_by_order[order]
            plt.fill(x1, y1,  alpha = alpha, c = color)
    # change node color for node 0
    
    nx.draw(G, pos, node_color = node_color, edge_color = link_color, with_labels = with_labels, width = link_width, node_size = node_size, ax = ax)


    