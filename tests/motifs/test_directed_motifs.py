from hypergraphx import DirectedHypergraph
from hypergraphx.motifs.directed_motifs import compute_directed_motifs


def test_compute_directed_motifs_order_three_no_null():
    """Test directed motif computation returns observed results only."""
    edges = [((0,), (1, 2)), ((1,), (0, 2)), ((2,), (0, 1))]
    hg = DirectedHypergraph(edge_list=edges)

    result = compute_directed_motifs(hg, order=3, runs_config_model=0)

    assert "observed" in result
    assert "config_model" not in result
    assert result["observed"]
