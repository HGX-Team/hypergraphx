from hypergraphx.motifs.utils import (
    diff_sum,
    norm_vector,
    avg,
    sigma,
    z_score,
    power_set,
    generate_motifs,
)


def test_diff_sum_and_norm_vector():
    """Test diff_sum and normalization helpers."""
    observed = [("a", 2), ("b", 4)]
    null_models = [[("a", 1), ("b", 3)], [("a", 2), ("b", 5)]]

    delta = list(diff_sum(observed, null_models))
    normed = norm_vector(delta)

    assert len(delta) == 2
    assert len(normed) == 2


def test_avg_sigma_z_score():
    """Test average, sigma, and z-score helpers."""
    motifs = [[("a", 1), ("b", 3)], [("a", 3), ("b", 5)]]

    u = avg(motifs)
    s = sigma(motifs)
    z = z_score(motifs[0], motifs)

    assert u == [2.0, 4.0]
    assert all(val >= 0 for val in s)
    assert len(z) == 2


def test_power_set_basic():
    """Test power set generation."""
    subsets = list(power_set([1, 2]))
    assert [] in subsets
    assert [1, 2] in subsets
    assert len(subsets) == 4


def test_generate_motifs_small():
    """Test motif generation for N=2 returns non-empty mapping."""
    mapping, labeling = generate_motifs(2)
    assert mapping
    assert labeling
