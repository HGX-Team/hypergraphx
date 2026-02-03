import numpy as np

from hypergraphx.generation.GAM import GroupAttractivenessModel


def test_gam_iteration_and_outputs():
    """Test GAM can run a single iteration and expose outputs."""
    model = GroupAttractivenessModel(n=5, balance=0.6, L=10, seed=0)
    model.run(1, max_edges=5, verbose=False)

    assert model.get_max_time() >= 0
    assert isinstance(model.get_attributes(), dict)


def test_gam_distance_periodic_box():
    """Test periodic box distance matrix shape."""
    rng = np.random.default_rng(1)
    points = rng.random((3, 2))
    dist = GroupAttractivenessModel.distance_in_a_periodic_box(points, boundary=1.0)
    assert dist.shape == (3, 3)
