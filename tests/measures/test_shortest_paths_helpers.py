import numpy as np

import networkx as nx

from hypergraphx import Hypergraph
from hypergraphx.measures.shortest_paths import (
    dict_to_df,
    HO_convert_node_labels_to_integers,
    calc_sizes_redundancies_of_shortest_paths,
    calc_prop_true_dyad_paths_per_spl,
    calc_prop_of_each_path_is_dyad,
)
from hypergraphx.representations.projections import clique_projection


def _make_hypergraph():
    edges = [(0, 1), (1, 2, 3)]
    return Hypergraph(edge_list=edges)


def test_dict_to_df_shapes():
    """Test conversion of shortest path dict to DataFrame."""
    hg = _make_hypergraph()
    g = clique_projection(hg, keep_isolated=True)
    sp_len = dict(nx.all_pairs_shortest_path_length(g))
    df = dict_to_df(sp_len, hg.get_nodes())
    assert df.shape == (hg.num_nodes(), hg.num_nodes())
    assert np.isnan(df.iloc[0, 0])


def test_convert_node_labels_to_integers():
    """Test relabeling to integer node labels."""
    hg = Hypergraph(edge_list=[("a", "b"), ("b", "c")])
    relabeled = HO_convert_node_labels_to_integers(hg)

    assert set(relabeled.get_nodes()) == set(range(relabeled.num_nodes()))
    assert len(relabeled.get_edges()) == len(hg.get_edges())


def test_calc_sizes_redundancies_and_props():
    """Test size/redundancy enrichment and derived proportions."""
    hg = _make_hypergraph()
    g = clique_projection(hg, keep_isolated=True)
    sp_dict = dict(nx.all_pairs_shortest_path(g))
    sp_len = dict(nx.all_pairs_shortest_path_length(g))
    spl = dict_to_df(sp_len, hg.get_nodes())

    enriched = calc_sizes_redundancies_of_shortest_paths(
        shortest_paths_ho=sp_dict,
        Hbase=hg,
        option="min",
        root=None,
    )

    sample = enriched[0][1]
    assert "sp" in sample
    assert "sizes" in sample
    assert "redundancies" in sample
    assert "avg_size" in sample
    assert len(sample["sizes"]) == len(sample["sp"]) - 1

    # Avoid mutating DataFrame .values directly (can be read-only under some pandas modes).
    avg_ord = np.full_like(spl.to_numpy(), 2.0)
    np.fill_diagonal(avg_ord, 0)
    avg_ord[0, 1] = 3
    avg_ord[1, 0] = 3
    # If only one boolean class appears, unstack drops the other column.
    props = calc_prop_true_dyad_paths_per_spl(spl.to_numpy(), avg_ord)
    assert "spl" in props.columns
    assert any(col in props.columns for col in ("prop_False", "prop_True"))

    prop_dyad = calc_prop_of_each_path_is_dyad(enriched, spl)
    assert {"spl", "pd"}.issubset(prop_dyad.columns)
