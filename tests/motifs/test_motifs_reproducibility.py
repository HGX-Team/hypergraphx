from hypergraphx import Hypergraph, DirectedHypergraph
from hypergraphx.motifs.motifs import compute_motifs
from hypergraphx.motifs.directed_motifs import compute_directed_motifs


def test_compute_motifs_reproducible_with_seed():
    hg = Hypergraph(edge_list=[(0, 1, 2), (1, 2, 3), (0, 2, 3)], weighted=False)
    out1 = compute_motifs(hg, order=3, runs_config_model=2, seed=0)
    out2 = compute_motifs(hg, order=3, runs_config_model=2, seed=0)
    assert out1 == out2


def test_compute_directed_motifs_reproducible_with_seed():
    dg = DirectedHypergraph(
        edge_list=[((0,), (1, 2)), ((1,), (2, 3)), ((2,), (0, 3))],
        weighted=False,
    )
    out1 = compute_directed_motifs(dg, order=3, runs_config_model=2, seed=0)
    out2 = compute_directed_motifs(dg, order=3, runs_config_model=2, seed=0)
    assert out1 == out2
