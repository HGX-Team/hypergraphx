import numpy as np

from hypergraphx.generation.hy_mmsbm_sampling import sample_truncated_poisson


def test_sample_truncated_poisson_positive():
    """Test truncated Poisson samples are positive."""
    rng = np.random.default_rng(0)
    samples = sample_truncated_poisson(np.array([0.5, 1.5, 3.0]), rng=rng)

    assert (samples > 0).all()
