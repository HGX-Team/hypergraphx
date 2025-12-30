from hypergraphx import MultiplexHypergraph
from hypergraphx.measures.multiplex.overlap import edge_overlap


def test_edge_overlap_across_layers():
    """Test overlap sums weights across layers."""
    hg = MultiplexHypergraph(weighted=True)
    hg.add_edge((0, 1), layer="L1", weight=1.0)
    hg.add_edge((0, 1), layer="L2", weight=2.5)

    assert edge_overlap(hg, (0, 1)) == 3.5
