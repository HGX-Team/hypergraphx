import pytest

from hypergraphx import Hypergraph
from hypergraphx.utils.cc import (
    connected_components,
    node_connected_component,
    num_connected_components,
    largest_component,
    largest_component_size,
    isolated_nodes,
    is_isolated,
    is_connected,
)


def _make_hypergraph():
    hg = Hypergraph(edge_list=[(0, 1), (2, 3)])
    hg.add_node(4)
    return hg


def test_connected_components_and_counts():
    """Test connected components and related helpers."""
    hg = _make_hypergraph()

    components = connected_components(hg)
    assert len(components) == 3
    assert num_connected_components(hg) == 3

    largest = largest_component(hg)
    assert len(largest) == 2
    assert largest_component_size(hg) == 2


def test_node_connected_component():
    """Test node component extraction."""
    hg = _make_hypergraph()
    comp = node_connected_component(hg, 0)
    assert set(comp) == {0, 1}


def test_isolated_nodes():
    """Test isolated node detection helpers."""
    hg = _make_hypergraph()

    assert isolated_nodes(hg) == [4]
    assert is_isolated(hg, 4) is True
    assert is_connected(hg) is False


def test_cc_invalid_args():
    """Test order/size validation."""
    hg = _make_hypergraph()
    with pytest.raises(ValueError, match="both specified"):
        connected_components(hg, order=1, size=2)
