import pytest
from collections import deque
from hypergraphx import Hypergraph
from hypergraphx.utils.visits import _bfs, _dfs

def test_bfs_basic():
    """Test BFS on a simple hypergraph with no max depth, order, or size constraints."""
    edge_list = [
        (1, 2, 3),  # Hyperedge connecting 1, 2, 3
        (2, 4),     # Hyperedge connecting 2, 4
    ]
    hg = Hypergraph(edge_list)
    result = _bfs(hg, start=1)
    expected = {1, 2, 3, 4}
    assert result == expected, f"Expected {expected}, got {result}"

def test_bfs_with_max_depth():
    """Test BFS with max depth constraint."""
    edge_list = [
        (1, 2, 3),    # Hyperedge connecting 1, 2, 3
        (2, 4, 5),    # Hyperedge connecting 2, 4, 5
        (5, 6),       # Hyperedge connecting 5, 6
    ]
    hg = Hypergraph(edge_list)
    result = _bfs(hg, start=1, max_depth=1)
    expected = {1, 2, 3}  # Only nodes within 1 depth from 1 should be visited
    assert result == expected, f"Expected {expected}, got {result}"

def test_dfs_basic():
    """Test DFS on a simple hypergraph with no max depth, order, or size constraints."""
    edge_list = [
        (1, 2, 3),  # Hyperedge connecting 1, 2, 3
        (2, 4),     # Hyperedge connecting 2, 4
    ]
    hg = Hypergraph(edge_list)
    result = _dfs(hg, start=1)
    expected = {1, 2, 3, 4}
    assert result == expected, f"Expected {expected}, got {result}"

def test_dfs_with_max_depth():
    """Test DFS with max depth constraint."""
    edge_list = [
        (1, 2, 3),    # Hyperedge connecting 1, 2, 3
        (2, 4, 5),    # Hyperedge connecting 2, 4, 5
        (5, 6),       # Hyperedge connecting 5, 6
    ]
    hg = Hypergraph(edge_list)
    result = _dfs(hg, start=1, max_depth=1)
    expected = {1, 2, 3}  # Only nodes within 1 depth from 1 should be visited
    assert result == expected, f"Expected {expected}, got {result}"

def test_bfs_with_order_and_size():
    """Test BFS with specific order and size constraints."""
    edge_list = [
        (1, 2),       # Hyperedge of order 1
        (1, 3, 4),    # Hyperedge of order 2
        (2, 5, 6),    # Hyperedge of order 2
        (3, 7, 8),    # Hyperedge of order 2
    ]
    hg = Hypergraph(edge_list)
    result = _bfs(hg, start=1, order=2)  # Only consider edges of order 2
    expected = {1, 3, 4, 7, 8}  # Nodes reachable through edges of order 2
    assert result == expected, f"Expected {expected}, got {result}"

def test_dfs_with_order_and_size():
    """Test DFS with specific order and size constraints."""
    edge_list = [
        (1, 2),       # Hyperedge of order 1
        (1, 3, 4),    # Hyperedge of order 2
        (2, 5, 6),    # Hyperedge of order 2
        (3, 7, 8),    # Hyperedge of order 2
    ]
    hg = Hypergraph(edge_list)
    result = _dfs(hg, start=1, order=2)  # Only consider edges of order 2
    expected = {1, 3, 4, 7, 8}  # Nodes reachable through edges of order 2
    assert result == expected, f"Expected {expected}, got {result}"

def test_bfs_empty_hypergraph():
    """Test BFS on an empty hypergraph, expecting ValueError due to missing start node."""
    hg = Hypergraph(edge_list=[])
    with pytest.raises(ValueError, match="Node 1 not in hypergraph."):
        _bfs(hg, start=1)

def test_dfs_empty_hypergraph():
    """Test DFS on an empty hypergraph, expecting ValueError due to missing start node."""
    hg = Hypergraph(edge_list=[])
    with pytest.raises(ValueError, match="Node 1 not in hypergraph."):
        _dfs(hg, start=1)
