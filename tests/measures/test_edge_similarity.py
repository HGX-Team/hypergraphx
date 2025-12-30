from hypergraphx.measures.edge_similarity import (
    intersection,
    jaccard_similarity,
    jaccard_distance,
)


def test_intersection_and_jaccard():
    """Test basic set similarity helpers."""
    a = {1, 2, 3}
    b = {2, 3, 4}

    assert intersection(a, b) == 2
    assert jaccard_similarity(a, b) == 0.5
    assert jaccard_distance(a, b) == 0.5
