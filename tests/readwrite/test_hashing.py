import pytest
from hypergraphx import (
    Hypergraph,
    DirectedHypergraph,
    TemporalHypergraph,
    MultiplexHypergraph,
)
from hypergraphx.readwrite.hashing import hash_hypergraph


@pytest.fixture
def sample_hypergraph():
    """Fixture for a basic Hypergraph."""
    hg = Hypergraph(
        edge_list=[("A", "B", "C"), ("C", "D")],
        weighted=True,
        weights=[1.5, 2.0],
        hypergraph_metadata={"description": "Test Hypergraph"},
        edge_metadata=[{"type": "triangle"}, {"type": "line"}],
        node_metadata={
            "A": {"color": "red"},
            "B": {"color": "blue"},
            "C": {"color": "green"},
            "D": {"color": "yellow"},
        },
    )
    return hg


@pytest.fixture
def sample_hypergraph_identical():
    """Fixture for an identical Hypergraph."""
    hg = Hypergraph(
        edge_list=[("A", "B", "C"), ("C", "D")],
        weighted=True,
        weights=[1.5, 2.0],
        hypergraph_metadata={"description": "Test Hypergraph"},
        edge_metadata=[{"type": "triangle"}, {"type": "line"}],
        node_metadata={
            "A": {"color": "red"},
            "B": {"color": "blue"},
            "C": {"color": "green"},
            "D": {"color": "yellow"},
        },
    )
    return hg


@pytest.fixture
def different_hypergraph():
    """Fixture for a different Hypergraph."""
    hg = Hypergraph(
        edge_list=[("A", "B"), ("B", "C")],
        weighted=False,
        hypergraph_metadata={"description": "Different Hypergraph"},
        edge_metadata=[{"type": "connection"}, {"type": "relation"}],
        node_metadata={
            "A": {"color": "red"},
            "B": {"color": "blue"},
            "C": {"color": "green"},
        },
    )
    return hg


@pytest.fixture
def sample_directed_hypergraph():
    """Fixture for a DirectedHypergraph."""
    dhg = DirectedHypergraph(
        edge_list=[(("A",), ("B",)), (("B",), ("C",))],
        weighted=True,
        weights=[1.5, 2.0],
        hypergraph_metadata={"description": "Directed Hypergraph"},
        edge_metadata=[{"type": "relation"}, {"type": "dependency"}],
        node_metadata={
            "A": {"role": "source"},
            "B": {"role": "middle"},
            "C": {"role": "target"},
        },
    )
    return dhg


@pytest.fixture
def sample_temporal_hypergraph():
    """Fixture for a TemporalHypergraph."""
    thg = TemporalHypergraph(
        edge_list=[("A", "B"), ("B", "C")],
        time_list=[10, 20],
        weighted=True,
        weights=[1.5, 2.0],
        hypergraph_metadata={"description": "Temporal Hypergraph"},
        edge_metadata=[
            {"type": "interaction", "event": "visit"},
            {"type": "interaction", "event": "call"},
        ],
        node_metadata={
            "A": {"status": "active"},
            "B": {"status": "inactive"},
            "C": {"status": "active"},
        },
    )
    return thg


@pytest.fixture
def sample_multiplex_hypergraph():
    """Fixture for a MultiplexHypergraph."""
    mhg = MultiplexHypergraph(
        edge_list=[("A", "B"), ("B", "C")],
        edge_layer=["social", "work"],
        weighted=True,
        weights=[1.5, 2.0],
        hypergraph_metadata={"description": "Multiplex Hypergraph"},
        edge_metadata=[{"type": "friendship"}, {"type": "colleague"}],
        node_metadata={
            "A": {"department": "HR"},
            "B": {"department": "Engineering"},
            "C": {"department": "Marketing"},
        },
    )
    return mhg


def test_identical_hypergraphs_same_hash(
    sample_hypergraph, sample_hypergraph_identical
):
    """Test that identical hypergraphs have the same hash."""
    hash1 = hash_hypergraph(sample_hypergraph)
    hash2 = hash_hypergraph(sample_hypergraph_identical)
    assert hash1 == hash2, "Identical hypergraphs should have the same hash."


def test_different_hypergraphs_different_hash(sample_hypergraph, different_hypergraph):
    """Test that different hypergraphs have different hashes."""
    hash1 = hash_hypergraph(sample_hypergraph)
    hash2 = hash_hypergraph(different_hypergraph)
    assert hash1 != hash2, "Different hypergraphs should have different hashes."


def test_hash_changes_with_node_metadata_change(sample_hypergraph):
    """Test that changing node metadata alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph.set_attr_to_node_metadata("A", "color", "purple")
    new_hash = hash_hypergraph(sample_hypergraph)
    assert (
        original_hash != new_hash
    ), "Hash should change when node metadata is modified."


def test_hash_changes_with_edge_metadata_change(sample_hypergraph):
    """Test that changing edge metadata alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    edge = ("A", "B", "C")
    sample_hypergraph.set_attr_to_edge_metadata(edge, "type", "updated_triangle")
    new_hash = hash_hypergraph(sample_hypergraph)
    assert (
        original_hash != new_hash
    ), "Hash should change when edge metadata is modified."


def test_hash_changes_with_edge_addition(sample_hypergraph):
    """Test that adding an edge alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph.add_edge(("D", "E"), weight=3.0, metadata={"type": "new_line"})
    new_hash = hash_hypergraph(sample_hypergraph)
    assert original_hash != new_hash, "Hash should change when a new edge is added."


def test_hash_changes_with_node_addition(sample_hypergraph):
    """Test that adding a node alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph.add_node("E", metadata={"color": "purple"})
    new_hash = hash_hypergraph(sample_hypergraph)
    assert original_hash != new_hash, "Hash should change when a new node is added."


def test_hash_temporal_hypergraph(sample_temporal_hypergraph):
    """Test hashing for TemporalHypergraph."""
    thg = sample_temporal_hypergraph
    thg_hash = hash_hypergraph(thg)
    assert (
        isinstance(thg_hash, str) and len(thg_hash) == 64
    ), "Hash should be a valid SHA-256 hex string."


def test_hash_multiplex_hypergraph(sample_multiplex_hypergraph):
    """Test hashing for MultiplexHypergraph."""
    mhg = sample_multiplex_hypergraph
    mhg_hash = hash_hypergraph(mhg)
    assert (
        isinstance(mhg_hash, str) and len(mhg_hash) == 64
    ), "Hash should be a valid SHA-256 hex string."


def test_hash_directed_hypergraph(sample_directed_hypergraph):
    """Test hashing for DirectedHypergraph."""
    dhg = sample_directed_hypergraph
    dhg_hash = hash_hypergraph(dhg)
    assert (
        isinstance(dhg_hash, str) and len(dhg_hash) == 64
    ), "Hash should be a valid SHA-256 hex string."


def test_hash_changes_after_removal(sample_hypergraph):
    """Test that removing a node alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph.remove_node("B", keep_edges=True)
    new_hash = hash_hypergraph(sample_hypergraph)
    assert original_hash != new_hash, "Hash should change when a node is removed."


def test_hash_unaffected_by_node_order_in_edges(sample_hypergraph):
    """Test that the hash is unaffected by node order in edges."""
    hg1 = Hypergraph(
        edge_list=[("A", "B", "C")],
        weighted=sample_hypergraph._weighted,
        weights=[1.5],
        hypergraph_metadata=sample_hypergraph._hypergraph_metadata,
        edge_metadata=[{"type": "triangle"}],
        node_metadata=sample_hypergraph._node_metadata,
    )
    hg2 = Hypergraph(
        edge_list=[("C", "B", "A")],
        weighted=sample_hypergraph._weighted,
        weights=[1.5],
        hypergraph_metadata=sample_hypergraph._hypergraph_metadata,
        edge_metadata=[{"type": "triangle"}],
        node_metadata=sample_hypergraph._node_metadata,
    )
    hash1 = hash_hypergraph(hg1)
    hash2 = hash_hypergraph(hg2)
    assert hash1 == hash2, "Hash should be the same regardless of node order in edges."


def test_hash_changes_with_edge_removal(sample_hypergraph):
    """Test that removing an edge alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    edge = ("A", "B", "C")
    sample_hypergraph.remove_edge(edge)
    new_hash = hash_hypergraph(sample_hypergraph)
    assert original_hash != new_hash, "Hash should change when an edge is removed."


def test_hash_changes_with_hypergraph_metadata_change(sample_hypergraph):
    """Test that changing hypergraph metadata alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph._hypergraph_metadata["description"] = "Modified Hypergraph"
    new_hash = hash_hypergraph(sample_hypergraph)
    assert (
        original_hash != new_hash
    ), "Hash should change when hypergraph metadata is modified."


def test_hash_empty_hypergraph():
    """Test hashing for an empty Hypergraph."""
    hg = Hypergraph()
    hg_hash = hash_hypergraph(hg)
    assert (
        isinstance(hg_hash, str) and len(hg_hash) == 64
    ), "Hash should be a valid SHA-256 hex string."


def test_hash_with_duplicaed_edge_unweighted_hypergraph():
    """Test hashing for a non-weighted Hypergraph with duplicated edges."""
    hg = Hypergraph(edge_list=[("A", "B"), ("C", "D")], weighted=False)
    hg_hash = hash_hypergraph(hg)
    hg.add_edge(("A", "B"))
    new_hash = hash_hypergraph(hg)
    assert (
        hg_hash == new_hash
    ), "Hash should be the same when adding a duplicate edge to a non-weighted Hypergraph."


def test_hash_non_weighted_hypergraph():
    """Test hashing for a non-weighted Hypergraph."""
    hg = Hypergraph(edge_list=[("A", "B"), ("B", "C")], weighted=False)
    hg_hash = hash_hypergraph(hg)
    assert (
        isinstance(hg_hash, str) and len(hg_hash) == 64
    ), "Hash should be a valid SHA-256 hex string."


def test_hash_changes_with_edge_weight_change(sample_hypergraph):
    """Test that changing an edge's weight alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    edge = ("A", "B", "C")
    sample_hypergraph.set_weight(edge, 3.0)
    new_hash = hash_hypergraph(sample_hypergraph)
    assert (
        original_hash != new_hash
    ), "Hash should change when an edge's weight is modified."


def test_hash_unaffected_by_edge_order(sample_hypergraph):
    """Test that the hash is unaffected by the order of edges."""
    # Create hypergraphs with edges in different orders
    hg1 = Hypergraph(
        edge_list=[("A", "B", "C"), ("C", "D")],
        weighted=sample_hypergraph._weighted,
        weights=[1.5, 2.0],
        hypergraph_metadata=sample_hypergraph._hypergraph_metadata,
    )
    hg2 = Hypergraph(
        edge_list=[("C", "D"), ("A", "B", "C")],
        weighted=sample_hypergraph._weighted,
        weights=[2.0, 1.5],  # Corresponding weights for the reordered edges
        hypergraph_metadata=sample_hypergraph._hypergraph_metadata,
    )
    hash1 = hash_hypergraph(hg1)
    hash2 = hash_hypergraph(hg2)
    assert hash1 == hash2, "Hash should be the same regardless of edge order."


def test_hash_different_node_identifiers(sample_hypergraph):
    """Test that hypergraphs with different node identifiers have different hashes."""
    hg1 = sample_hypergraph
    # Create a new hypergraph with the same structure but different node identifiers
    node_mapping = {"A": "X", "B": "Y", "C": "Z", "D": "W"}
    edge_list = [tuple(node_mapping[node] for node in edge) for edge in hg1.get_edges()]
    edge_metadata_list = list(hg1._edge_metadata.values())
    node_metadata = {
        node_mapping[node]: data for node, data in hg1._node_metadata.items()
    }
    hg2 = Hypergraph(
        edge_list=edge_list,
        node_metadata=node_metadata,
        edge_metadata=edge_metadata_list,
        weighted=hg1._weighted,
        weights=[1.5, 2.0],
        hypergraph_metadata=hg1._hypergraph_metadata,
    )
    hash1 = hash_hypergraph(hg1)
    hash2 = hash_hypergraph(hg2)
    assert hash1 != hash2, "Hashes should differ when node identifiers are different."


def test_hash_with_complex_metadata():
    """Test hashing for a hypergraph with complex metadata."""
    hg = Hypergraph(
        edge_list=[("A", "B")],
        node_metadata={
            "A": {"attributes": {"age": 25, "status": "active"}},
            "B": {"attributes": {"age": 30, "status": "inactive"}},
        },
        edge_metadata=[{"details": {"created": "2021-01-01", "active": True}}],
        hypergraph_metadata={
            "description": "Complex Metadata Hypergraph",
            "version": 1.0,
            "tags": ["test", "complex"],
            "nested": {"key1": "value1", "key2": [1, 2, 3]},
        },
        weighted=False,
    )
    hg_hash = hash_hypergraph(hg)
    assert (
        isinstance(hg_hash, str) and len(hg_hash) == 64
    ), "Hash should be a valid SHA-256 hex string."


def test_hash_changes_with_nested_metadata_change():
    """Test that changing nested metadata alters the hash."""
    hg = Hypergraph(
        edge_list=[("A", "B")],
        node_metadata={
            "A": {"attributes": {"age": 25, "status": "active"}},
            "B": {"attributes": {"age": 30, "status": "inactive"}},
        },
        edge_metadata=[{"details": {"created": "2021-01-01", "active": True}}],
        hypergraph_metadata={
            "description": "Complex Metadata Hypergraph",
            "version": 1.0,
            "tags": ["test", "complex"],
            "nested": {"key1": "value1", "key2": [1, 2, 3]},
        },
        weighted=False,
    )
    original_hash = hash_hypergraph(hg)
    # Modify nested metadata
    hg._hypergraph_metadata["nested"]["key2"].append(4)
    new_hash = hash_hypergraph(hg)
    assert (
        original_hash != new_hash
    ), "Hash should change when nested metadata is modified."


def test_hash_unaffected_by_non_hashable_attributes(sample_hypergraph):
    """Test that adding non-hashable attributes does not affect the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph.some_temporary_data = [
        1,
        2,
        3,
    ]  # Attribute not included in hashing
    new_hash = hash_hypergraph(sample_hypergraph)
    assert (
        original_hash == new_hash
    ), "Hash should be unaffected by non-hashable attributes."


def test_hash_consistency_across_runs(sample_hypergraph):
    """Test that the hash remains consistent across multiple runs."""
    hash1 = hash_hypergraph(sample_hypergraph)
    hash2 = hash_hypergraph(sample_hypergraph)
    assert hash1 == hash2, "Hash should remain consistent across multiple computations."


def test_hash_changes_with_node_removal(sample_hypergraph):
    """Test that removing a node alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph.remove_node("C", keep_edges=False)
    new_hash = hash_hypergraph(sample_hypergraph)
    assert original_hash != new_hash, "Hash should change when a node is removed."


def test_hash_after_node_metadata_removal(sample_hypergraph):
    """Test that removing node metadata alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    sample_hypergraph.set_node_metadata("A", {})
    new_hash = hash_hypergraph(sample_hypergraph)
    assert (
        original_hash != new_hash
    ), "Hash should change when node metadata is removed."


def test_hash_after_edge_metadata_removal(sample_hypergraph):
    """Test that removing edge metadata alters the hash."""
    original_hash = hash_hypergraph(sample_hypergraph)
    edge = ("A", "B", "C")
    sample_hypergraph.set_edge_metadata(edge, {})
    new_hash = hash_hypergraph(sample_hypergraph)
    assert (
        original_hash != new_hash
    ), "Hash should change when edge metadata is removed."


def test_hash_after_changing_edge_node_order(sample_hypergraph):
    """Test that changing node order within an edge does not affect the hash."""
    hg = sample_hypergraph
    original_hash = hash_hypergraph(hg)
    # Change node order in an edge
    edge = ("A", "B", "C")
    sorted_edge = tuple(sorted(edge))
    edge_id = hg._edge_list[sorted_edge]
    # Remove and re-add edge with nodes in different order
    hg.remove_edge(edge)
    hg.add_edge(("C", "A", "B"), weight=1.5, metadata={"type": "triangle"})
    new_hash = hash_hypergraph(hg)
    assert (
        original_hash == new_hash
    ), "Hash should be unaffected by node order within edges."
