import matplotlib.pyplot as plt
from hnx.representations.projections import clique_projection
import networkx as nx
import numpy as np

def draw_communities(
    H,
    figsize = (12, 6),
    ns = 50,
    mrk = 'o',
    edgecolor_node = 'grey',
    edgecolor_edge = 'lightgrey',
    overlapping = True,
    radius = 0.005
):

    plt.figure(figsize=figsize)
    wedgeprops = {'edgecolor': edgecolor_edge}

    G = clique_projection(H)
    pos = nx.spring_layout(G, k=0.1, seed=0)
    degree = dict(G.degree())

    if overlapping:
        plt.subplot(1, 2, 1)
        ax = plt.gca()
        nx.draw_networkx_edges(G, pos, arrows=False, edge_color=edgecolor_edge)
        for n, d in G.nodes(data=True):
            wedge_sizes, wedge_colors = viz.extract_bridge_properties(vcm.nodeName2Id[n], cm, vcm.U['gt'], threshold=0.01)
            if len(wedge_sizes) > 0:
                pie, t = plt.pie(wedge_sizes, center=pos[n], colors=wedge_colors, radius=(min(10, degree[n])) * radius,
                                 wedgeprops=wedgeprops)
                ax.axis("equal")
        plt.tight_layout()
        plt.title('Metadata')

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        nx.draw_networkx_edges(G, pos, arrows=False, edge_color=edgecolor_edge)
        for n, d in G.nodes(data=True):
            wedge_sizes, wedge_colors = viz.extract_bridge_properties(vcm.nodeName2Id[n], cm, vcm.U['HyMT'], threshold=0.01)
            if len(wedge_sizes) > 0:
                pie, t = plt.pie(wedge_sizes, center=pos[n], colors=wedge_colors, radius=(min(10, degree[n])) * radius,
                                 wedgeprops=wedgeprops, normalize=True)
                ax.axis("equal")
        plt.tight_layout()
        # plt.title('Hypergraph-MT')

    else:
        plt.subplot(1, 2, 1)
        nx.draw_networkx_edges(G, pos, arrows=False, edge_color=edgecolor_edge)
        for n, d in G.nodes(data=True):
            nx.draw_networkx_nodes(G, pos, [n], node_size=ns, node_shape=mrk, node_color=[int(d['node_color_gt'])],
                                   edgecolors=edgecolor_node, cmap=cmap, vmin=0, vmax=cmax)
        plt.title('Metadata')

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        nx.draw_networkx_edges(G, pos, arrows=False, edge_color=edgecolor_edge)
        for nid, n in enumerate(list(G.nodes())):
            nx.draw_networkx_nodes(G, pos, [n], node_size=ns, node_shape=mrk, node_color=[np.argmax(vcm.U['HyMT'][nid])],
                                   edgecolors=edgecolor_node, cmap=cmap, vmin=0, vmax=cmax)
        plt.title('Hypergraph-MT')