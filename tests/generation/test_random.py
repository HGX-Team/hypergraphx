import pytest
import random
from hypergraphx.generation.random import random_shuffle
from hypergraphx import Hypergraph


@pytest.fixture
def dummy_hypergraph():
    edges = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    return Hypergraph(edges)


def test_no_shuffle(dummy_hypergraph):
    """Test that with p=0, the hypergraph remains unchanged."""
    random.seed(42)
    hg = dummy_hypergraph
    original_edges = list(hg.get_edges(3))
    random_shuffle(hg, size=3, inplace=True, p=0.0)
    new_edges = list(hg.get_edges(3))
    assert (
        new_edges == original_edges
    ), f"Expected edges unchanged for p=0, got {new_edges}"


def test_full_shuffle():
    """Test that with p=1, all hyperedges are replaced.

    Using a fresh instance here to ensure no interference from other tests.
    """
    random.seed(42)
    edges = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    hg = Hypergraph(edges)
    original_edges = list(hg.get_edges(size=3))
    random_shuffle(hg, size=3, inplace=True, p=1.0)
    new_edges = list(hg.get_edges(size=3))
    # With p=1.0, all edges should be replaced with new random ones.
    assert (
        new_edges != original_edges
    ), f"Expected edges shuffled for p=1, got {new_edges}"
    assert len(new_edges) == len(
        original_edges
    ), "Number of edges should remain the same"


def test_intermediate_shuffle():
    """Test that with p=0.5, exactly half of the hyperedges are replaced.

    Using a fresh instance to ensure no interference from other tests.
    """
    import random

    random.seed(42)
    edges = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    hg = Hypergraph(edges)
    original_edges = list(hg.get_edges(size=3))

    p = 0.5
    random_shuffle(hg, size=3, inplace=True, p=p)
    new_edges = list(hg.get_edges(size=3))

    # Ensure the number of edges remains the same.
    assert len(new_edges) == len(
        original_edges
    ), "Number of edges should remain the same"

    # Count how many hyperedges were replaced.
    num_replaced = sum(1 for orig, new in zip(original_edges, new_edges) if orig != new)
    expected_replacements = int(p * len(original_edges))

    assert (
        num_replaced == expected_replacements
    ), f"Expected {expected_replacements} hyperedges replaced, got {num_replaced}"


def test_non_inplace(dummy_hypergraph):
    """Test that non-inplace operation returns a new hypergraph while leaving the original unchanged."""
    random.seed(42)
    hg = dummy_hypergraph
    original_edges = list(hg.get_edges(size=3))
    new_hg = random_shuffle(hg, size=3, inplace=False, p=1.0)
    # Original hypergraph should remain unchanged.
    assert (
        list(hg.get_edges(size=3)) == original_edges
    ), "Original hypergraph should remain unchanged in non-inplace mode"
    # The new hypergraph should have different edges.
    assert (
        list(new_hg.get_edges(size=3)) != original_edges
    ), "New hypergraph should have shuffled edges"


def test_both_order_and_size_error(dummy_hypergraph):
    """Test that specifying both order and size raises a ValueError."""
    hg = dummy_hypergraph
    with pytest.raises(ValueError, match="both specified"):
        random_shuffle(hg, order=2, size=3, inplace=True, p=1.0)


def test_neither_order_nor_size_error(dummy_hypergraph):
    """Test that specifying neither order nor size raises a ValueError."""
    hg = dummy_hypergraph
    with pytest.raises(ValueError, match="must be specified"):
        random_shuffle(hg, inplace=True, p=1.0)


def test_invalid_p_error(dummy_hypergraph):
    """Test that an invalid value of p raises a ValueError."""
    hg = dummy_hypergraph
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        random_shuffle(hg, size=3, inplace=True, p=1.5)
