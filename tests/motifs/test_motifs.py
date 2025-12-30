from hypergraphx import Hypergraph
from hypergraphx.motifs.motifs import compute_motifs


def test_compute_motifs_order_three_no_null():
    """Test motif computation returns observed results only when runs_config_model=0."""
    hg = Hypergraph(edge_list=[(0, 1), (1, 2), (0, 1, 2)])
    result = compute_motifs(hg, order=3, runs_config_model=0)

    assert "observed" in result
    assert "config_model" not in result
    assert result["observed"]
