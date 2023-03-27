import numpy as np
import pytest

from hypergraphx import Hypergraph
from hypergraphx.measures.sub_hypergraph_centrality import subhypergraph_centrality


# Fixture loaded_hypergraph defined inside the package-level conftest.py


@pytest.fixture
def hypergraph_with_sub_hc(loaded_hypergraph: Hypergraph):
    sub_hc = subhypergraph_centrality(loaded_hypergraph)
    return loaded_hypergraph, sub_hc


def test_sub_hc_type(hypergraph_with_sub_hc):
    _, sub_hc = hypergraph_with_sub_hc
    assert isinstance(sub_hc, np.ndarray)


def test_sub_hc_shape(hypergraph_with_sub_hc):
    hypergraph, sub_hc = hypergraph_with_sub_hc
    assert sub_hc.shape == (hypergraph.num_nodes(),)
