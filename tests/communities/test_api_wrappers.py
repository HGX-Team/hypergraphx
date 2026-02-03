import warnings

from hypergraphx import Hypergraph
from hypergraphx.communities.api import fit_hysc, run_core_periphery


def test_run_core_periphery_result_shape():
    hg = Hypergraph(edge_list=[(0, 1, 2), (1, 2, 3), (3, 4)], weighted=False)
    res = run_core_periphery(hg, n_iter=2, seed=0)
    assert isinstance(res.scores, dict)
    assert set(res.scores.keys()) == set(hg.get_nodes())


def test_fit_hysc_returns_labels():
    warnings.filterwarnings(
        "ignore",
        message="Could not find the number of physical cores*",
        category=UserWarning,
    )
    hg = Hypergraph(edge_list=[(0, 1), (1, 2), (2, 3)], weighted=False)
    res = fit_hysc(hg, k=2, seed=0)
    assert res.memberships.shape[0] == hg.num_nodes()
    assert res.labels.shape[0] == hg.num_nodes()
