from hypergraphx import Hypergraph

import matplotlib.pyplot as plt
import networkx as nx
import itertools
from matplotlib.patches import Polygon

def draw_motifs(patterns, 
                edge_size_colors=None,
                node_labels=None,
                node_size=500,
                node_color='lightblue',
                edge_color='black',
                save_path=None):
    # Collect all unique nodes across all patterns
    all_nodes = set(itertools.chain.from_iterable(itertools.chain.from_iterable(patterns)))
    G_global = nx.Graph()
    G_global.add_nodes_from(all_nodes)
    global_pos = nx.spring_layout(G_global, seed=42)  # consistent layout

    if edge_size_colors is None:
        edge_size_colors = {
            3: '#FFDAB9',  # light orange
            4: '#ADD8E6'   # light blue
        }
    
    default_color = '#D3D3D3'  # light gray for other sizes
        
    edge_sizes = set(len(edge) for graph in patterns for edge in graph if len(edge) > 2)
    for size in edge_sizes:
        if size not in edge_size_colors:
            edge_size_colors[size] = default_color

    # Set up plots
    num_graphs = len(patterns)
    fig, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))
    if num_graphs == 1:
        axes = [axes]

    # Plot each hypergraph
    for idx, (hypergraph, ax) in enumerate(zip(patterns, axes)):
        G = nx.Graph()
        nodes = set(itertools.chain.from_iterable(hypergraph))
        G.add_nodes_from(nodes)
        pos = {n: global_pos[n] for n in nodes}

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, edgecolors="black")

        if node_labels:        
            nx.draw_networkx_labels(G, pos, ax=ax)

        # Draw hyperedges
        for i, hedge in enumerate(hypergraph):
            hedge_pos = [pos[n] for n in hedge]
            edge_size = len(hedge)

            if edge_size < 2:
                continue  # skip size-1

            if edge_size == 2:
                # Draw as traditional edge
                nx.draw_networkx_edges(G, pos, edgelist=[tuple(hedge)], ax=ax, edge_color=edge_color, width=2)
            else:
                color = edge_size_colors[edge_size]
                polygon = Polygon(hedge_pos, closed=True, fill=True, alpha=0.3, color=color, edgecolor=edge_color)
                ax.add_patch(polygon)

        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
