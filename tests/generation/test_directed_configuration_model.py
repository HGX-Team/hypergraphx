import random

from hypergraphx import DirectedHypergraph
from hypergraphx.generation.directed_configuration_model import (
    directed_configuration_model,
)


def test_directed_configuration_model_preserves_sizes():
    """Test directed configuration model preserves edge sizes and count."""
    random.seed(0)
    edges = [((0,), (1, 2)), ((2,), (0,))]
    hg = DirectedHypergraph(edge_list=edges)

    sampled = directed_configuration_model(hg)

    assert len(sampled.get_edges()) == len(edges)
    assert sorted(len(e[0]) for e in sampled.get_edges()) == [1, 1]
    assert sorted(len(e[1]) for e in sampled.get_edges()) == [1, 2]
