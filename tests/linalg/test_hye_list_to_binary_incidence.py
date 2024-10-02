import numpy as np
import pytest
from scipy import sparse

from hypergraphx.linalg.linalg import hye_list_to_binary_incidence

hye_lists = [
    [
        (0, 1, 2),
        (1, 2),
        (0, 1),
        (3, 0, 2),
    ],
    [
        (0, 1, 2, 3, 4, 5, 6),
        (1, 2),
        (0, 1),
        (2, 5, 1),
        (2, 5, 3),
    ],
]
hye_lists_with_repetitions_and_null = [
    [
        (0, 1),
        (1, 2),
        (0, 1),
        (3, 0, 2),
    ],
    [
        (0, 1),
        (1, 2),
        (1, 0),
        (3, 0, 2),
    ],
    [
        (0,),
        (),
        (0, 1, 2),
        (3,),
        (),
    ],
    [
        (0,),
        (),
        (0, 1, 2),
        (3,),
        (),
        (0, 1, 2),
        (),
        (),
        (),
    ],
    [
        (0, 1, 2, 3, 4, 5, 6),
        (1, 2),
        (0, 1),
        (2, 5, 1),
        (2, 1, 5),
        (2, 5, 3),
    ],
    [
        (0, 1, 1, 1, 1, 1, 2),
        (1, 2, 2, 1),
        (0, 1),
        (2, 5, 1),
        (2, 1, 5),
        (2, 5, 3),
    ],
]
shapes = [
    None,
    (4, 4),
    (4, 5),
    (4, 9),
    (4, 1000),
    (100, 4),
    (100, 5),
]


def is_fail_config(hye_list, shape):
    N = max(map(max, (hye for hye in hye_list if hye))) + 1
    E = len(hye_list)

    return shape is not None and (shape[0] < N or shape[1] < E)


failing_configs = [
    (hye_list, shape)
    for hye_list in hye_lists
    for shape in shapes
    if is_fail_config(hye_list, shape)
]
correct_configs = [
    (hye_list, shape)
    for hye_list in hye_lists
    for shape in shapes
    if not is_fail_config(hye_list, shape)
]


@pytest.fixture(scope="class")
def expected_shape(hye_list, shape):
    N = max(map(max, (hye for hye in hye_list if hye))) + 1
    E = len(hye_list)

    not_valid = shape is not None and (shape[0] < N or shape[1] < E)
    if not_valid:
        return

    if shape is not None:
        expected_shape = shape
    else:
        expected_shape = (N, E)

    return expected_shape


@pytest.mark.parametrize("hye_list,shape", correct_configs, scope="class")
class TestHyeListToBinaryIncidenceFromListCorrect:
    def test_incidence_type(self, hye_list, shape, expected_shape):
        sparse_incidence = hye_list_to_binary_incidence(hye_list, shape)
        print(type(sparse_incidence))
        assert isinstance(sparse_incidence, sparse.coo_array)

    def test_incidence_shape(self, hye_list, shape, expected_shape):
        sparse_incidence = hye_list_to_binary_incidence(hye_list, shape)
        assert sparse_incidence.shape == expected_shape

    def test_incidence_only_contains_ones(self, hye_list, shape, expected_shape):
        sparse_incidence = hye_list_to_binary_incidence(hye_list, shape)
        assert np.all(sparse_incidence.data == 1)

    def test_with_dense(self, hye_list, shape, expected_shape):
        sparse_incidence = hye_list_to_binary_incidence(hye_list, shape)

        dense = np.zeros(expected_shape)
        for idx, hye in enumerate(hye_list):
            dense[hye, idx] = 1
        assert np.all(sparse_incidence.todense() == dense)


@pytest.mark.parametrize("hye_list,shape", failing_configs)
def test_raises_error(hye_list, shape):
    N = max(map(max, (hye for hye in hye_list if hye))) + 1
    E = len(hye_list)

    if shape is not None:
        if shape[0] < N or shape[1] < E:
            with pytest.raises(ValueError):
                hye_list_to_binary_incidence(hye_list, shape)
