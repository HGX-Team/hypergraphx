import numpy as np
import pytest

from hypergraphx.utils.community import normalize_array, calculate_permutation_matrix


def test_normalize_array_handles_zero_rows():
    """Test normalization does not divide by zero for empty rows."""
    u = np.array([[1.0, 1.0], [0.0, 0.0]])
    normalized = normalize_array(u, axis=1)

    assert np.allclose(normalized[0], [0.5, 0.5])
    assert np.allclose(normalized[1], [0.0, 0.0])


def test_calculate_permutation_matrix_swapped_columns():
    """Test permutation matrix swaps columns to match reference."""
    u_ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    u_pred = np.array([[0.0, 1.0], [1.0, 0.0]])

    P = calculate_permutation_matrix(u_ref, u_pred)
    assert np.allclose(u_pred @ P, u_ref)


def test_calculate_permutation_matrix_shape_mismatch():
    """Test permutation matrix raises on shape mismatch."""
    u_ref = np.zeros((2, 2))
    u_pred = np.zeros((3, 2))

    with pytest.raises(ValueError, match="same shape"):
        calculate_permutation_matrix(u_ref, u_pred)
