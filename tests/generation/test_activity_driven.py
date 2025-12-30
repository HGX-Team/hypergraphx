import random

from hypergraphx.generation.activity_driven import HOADmodel


def test_hoadmodel_generates_temporal_hypergraph():
    """Test HOADmodel returns a temporal hypergraph with valid timestamps."""
    random.seed(0)
    activities = {2: [1.0, 0.0, 0.0]}
    thg = HOADmodel(N=3, activities_per_order=activities, time=2)

    times = [t for t, _ in thg.get_edges()]
    assert all(t in (0, 1) for t in times)
