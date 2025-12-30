import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hypergraphx import Hypergraph
from hypergraphx.viz.draw_hypergraph import draw_hypergraph
from hypergraphx.viz.draw_projections import draw_bipartite, draw_clique
from hypergraphx.viz.draw_simplicial import draw_SC
from hypergraphx.viz.draw_motifs import draw_motifs
from hypergraphx.viz.plot_motifs import plot_motifs


def _make_hypergraph():
    return Hypergraph(edge_list=[(0, 1), (1, 2, 3)])


def test_draw_hypergraph_smoke(monkeypatch):
    """Test draw_hypergraph runs without errors."""
    monkeypatch.setattr(plt, "show", lambda: None)
    hg = _make_hypergraph()

    draw_hypergraph(hg)


def test_draw_projections(monkeypatch):
    """Test draw_bipartite and draw_clique return axes."""
    monkeypatch.setattr(plt, "show", lambda: None)
    hg = _make_hypergraph()

    ax_bi = draw_bipartite(hg)
    ax_cl = draw_clique(hg)

    assert ax_bi is not None
    assert ax_cl is not None


def test_draw_simplicial(monkeypatch):
    """Test draw_SC runs without errors."""
    monkeypatch.setattr(plt, "show", lambda: None)
    hg = _make_hypergraph()

    draw_SC(hg)


def test_draw_motifs_and_plot(tmp_path, monkeypatch):
    """Test motif plotting utilities."""
    monkeypatch.setattr(plt, "show", lambda: None)
    patterns = [[(0, 1), (1, 2, 3)]]
    draw_motifs(patterns, save_path=str(tmp_path / "motifs.png"))

    plot_motifs([0.1, -0.2, 0.3, -0.1, 0.05, 0.2], save_name=str(tmp_path / "plot.png"))
