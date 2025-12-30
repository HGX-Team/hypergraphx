import pandas as pd

from hypergraphx import Hypergraph
from hypergraphx.filters import statistical_filters
from hypergraphx.filters.statistical_filters import get_svh, get_svc


class DummyPool:
    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return list(map(func, iterable))

    def close(self):
        return None


def _make_hypergraph():
    edges = [(0, 1), (1, 2), (0, 1, 2)]
    return Hypergraph(edge_list=edges)


def test_get_svh_basic():
    """Test SVH extraction returns a dictionary of DataFrames."""
    hg = _make_hypergraph()

    svh = get_svh(hg, max_order=3, mp=False)

    assert isinstance(svh, dict)
    assert all(isinstance(df, pd.DataFrame) for df in svh.values())


def test_get_svc_with_dummy_pool(monkeypatch):
    """Test SVC extraction with a dummy pool to avoid multiprocessing."""
    hg = _make_hypergraph()

    monkeypatch.setattr(statistical_filters, "Pool", DummyPool)
    monkeypatch.setattr(statistical_filters, "cpu_count", lambda: 1)

    svc = get_svc(hg, min_order=2, max_order=3)

    assert isinstance(svc, pd.DataFrame)
    assert {"group", "pvalue", "w", "fdr"}.issubset(svc.columns)
