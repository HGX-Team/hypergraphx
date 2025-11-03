from hypergraphx.core.Hypergraph import Hypergraph
from hypergraphx.viz.HypergraphVisualizer import HypergraphVisualizer

def draw_hypergraph(hypergraph: Hypergraph, *args, **kwargs):
    """Visualize a hypergraph."""
    # Initialize HypergraphVisualizer object
    hypergraph_visualizer = HypergraphVisualizer(hypergraph)
    hypergraph_visualizer.draw(*args, **kwargs)