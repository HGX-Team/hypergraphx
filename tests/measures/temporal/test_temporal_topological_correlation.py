import networkx as nx
import pytest

pytest.importorskip("tqdm")

from hypergraphx import Hypergraph, TemporalHypergraph
from hypergraphx.measures.temporal.temporal_topological_correlation import (
    _to_df,
    compute_all_nodes_shortest_path,
    compute_all_edges_shortest_path,
    get_mean_distance_events,
)
from hypergraphx.representations.projections import clique_projection


def test_to_df_temporal_hypergraph():
    """Test conversion of temporal hypergraph to DataFrame."""
    thg = TemporalHypergraph(edge_list=[(0, (0, 1)), (1, (1, 2, 3))])
    df = _to_df(thg)

    assert set(df.columns) == {"timestamp", "nodes", "order"}
    assert df["order"].tolist() == [2, 3]


def test_compute_all_nodes_shortest_path():
    """Test node shortest path distances for connected graph."""
    G = nx.path_graph(3)
    dist = compute_all_nodes_shortest_path(G)

    assert dist[0][2] == 2


def test_compute_all_edges_shortest_path_aggregate():
    """Test edge distance computation on an aggregated hypergraph."""
    hg = Hypergraph(edge_list=[(0, 1), (1, 2)])
    graph_distance = compute_all_nodes_shortest_path(clique_projection(hg))

    distances = compute_all_edges_shortest_path(
        hg, graph_distance=graph_distance, aggregate=True
    )

    assert distances[(0, 1), (0, 1)] == 0
    assert distances[(0, 1), (1, 2)] == 1


def test_get_mean_distance_events_same_order():
    """Test mean distance counts for same-order temporal events."""
    thg = TemporalHypergraph(edge_list=[(0, (0, 1)), (1, (1, 2))])
    edge1 = (0, (0, 1))
    edge2 = (1, (1, 2))
    distances = {
        (edge1, edge1): 0,
        (edge2, edge2): 0,
        (edge1, edge2): 1,
        (edge2, edge1): 1,
    }

    counts = get_mean_distance_events(thg, order=2, edge_distance=distances)
    assert sum(counts.values()) == 1
