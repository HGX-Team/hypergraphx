import matplotlib.pyplot as plt
import networkx as nx

from hypergraphx.representations.projections import clique_projection


def find_triplets(list):
    triplets = []
    for i in range(len(list)):
        for j in range(i+1, len(list)):
            for k in range(j+1, len(list)):
                triplets.append([list[i], list[j], list[k]])
    return triplets

def draw_SC(HG, pos = None, link_color = 'black',
             hyperlink_color_by_order = {2:'r', 3:'orange', 4:'green'}, 
             link_width = 2, node_size = 150,
             node_color = '#5494DA',
             with_labels = False,
             ax = None):
    
    G = clique_projection(HG, keep_isolated=True)
    if pos == None:
        pos = nx.spring_layout(G)
    for h_edge in HG.get_edges():
        if len(h_edge) > 2:
            
            order = len(h_edge)-1
            
            if order >= 5:
                alpha = .1
            else:
                alpha = .5
            
            if order not in hyperlink_color_by_order.keys():
                hyperlink_color_by_order[order] = 'Black'
            color = hyperlink_color_by_order[order]

                
            x_coor = []
            y_coor = []
            triplets = find_triplets(h_edge)
            for triplet in triplets:
                for node in triplet:
                    x_coor.append(pos[node][0])
                    y_coor.append(pos[node][1])
                #print(triplet)


                plt.fill(x_coor, y_coor,  alpha = alpha, c = color)

    nx.draw(G, pos =pos, with_labels = with_labels, node_color = node_color, edge_color = link_color, width = link_width, node_size = node_size, ax= ax)