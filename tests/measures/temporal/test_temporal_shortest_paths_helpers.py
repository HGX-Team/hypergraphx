import networkx as nx
import pytest

pytest.importorskip("tqdm")
pytest.importorskip("requests")

from hypergraphx import Hypergraph
from hypergraphx.measures.temporal.temporal_shortest_paths import (
    get_ds_windowsize,
    relabel,
    supra_adj,
    calc_size_of_single_path,
    P4_calc_shortest_fastest_paths,
)


def test_get_ds_windowsize():
    """Test window size selection helper."""
    assert get_ds_windowsize(100) == 100
    assert get_ds_windowsize(600) == 300


def test_relabel_edges():
    """Test edge relabeling in temporal shortest paths module."""
    edges = [("a", "b"), ("b", "c")]
    mapping = {"a": 0, "b": 1, "c": 2}
    relabeled = relabel(edges, mapping)

    assert relabeled == [(0, 1), (1, 2)]


def test_supra_adj_graph():
    """Test supra-adjacency construction for temporal graphs."""
    g0 = nx.Graph()
    g0.add_edge(1, 2)
    g1 = nx.Graph()
    g1.add_edge(1, 2)
    temporal = {0: g0, 1: g1}

    G = supra_adj(temporal, [0, 1], {1, 2, 3}, dataset="toy")

    assert (0, 1) in G.nodes
    assert (1, 1) in G.nodes
    assert G.has_edge((0, 1), (1, 1))


def test_calc_size_of_single_path():
    """Test size and redundancy extraction for a simple path."""
    h0 = Hypergraph(edge_list=[(0, 1)])
    temporal = {0: h0, 1: h0}
    hyperpath = [(0, 0), (0, 1), (1, 1)]

    sizes, timesteps, redundancies = calc_size_of_single_path(
        hyperpath, temporal, dataset="toy", option="min"
    )

    assert sizes[0] == 2
    assert len(timesteps) == 2
    assert len(redundancies) == 2


def test_p4_calc_shortest_fastest_paths():
    """Test shortest/fastest path calculations on a tiny supra graph."""
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=1)

    fdict, sdict = P4_calc_shortest_fastest_paths([0], G, verbose=False)

    assert 0 in fdict
    assert 0 in sdict
