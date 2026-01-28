import pytest

from hypergraphx import Hypergraph
from hypergraphx.exceptions import InvalidParameterError
from hypergraphx.measures.reducibility import reducibility, layer_reducibility


def _small_hypergraph():
    hg = Hypergraph()
    hg.add_nodes([0, 1, 2, 3])
    hg.add_edges([(0, 1), (1, 2), (2, 3), (0, 1, 2)])
    return hg


def _simplex_hypergraph():
    hg = Hypergraph()
    hg.add_nodes([0, 1, 2])
    hg.add_edges([(0, 1), (0, 2), (1, 2), (0, 1, 2)])
    return hg


def test_reducibility_basic():
    hg = _small_hypergraph()
    eta, reps = reducibility(hg, optimization="exact", entropy_method="count")
    assert 0.0 <= eta <= 1.0
    assert max(reps) == max(hg.get_sizes())


def test_layer_reducibility_basic():
    hg = _small_hypergraph()
    etas = layer_reducibility(hg, entropy_method="count")
    assert set(etas.keys()) == set(hg.get_sizes())
    for value in etas.values():
        assert 0.0 <= value <= 1.0


def test_reducibility_partition_dict():
    hg = _small_hypergraph()
    partition = {0: 0, 1: 0, 2: 1, 3: 1}
    eta, reps = reducibility(hg, partition=partition, entropy_method="count")
    assert 0.0 <= eta <= 1.0
    assert max(reps) == max(hg.get_sizes())


def test_reducibility_greedy():
    hg = _small_hypergraph()
    eta, reps = reducibility(hg, optimization="greedy", entropy_method="count")
    assert 0.0 <= eta <= 1.0
    assert max(reps) == max(hg.get_sizes())


def test_reducibility_matches_known_value():
    hg = _small_hypergraph()
    eta, reps = reducibility(hg, entropy_method="count", optimization="exact")
    assert eta == pytest.approx(0.2665484173158311)
    assert reps == (3,)


def test_layer_reducibility_known_values():
    hg = _small_hypergraph()
    etas = layer_reducibility(hg, entropy_method="count")
    assert etas[2] == pytest.approx(0.2665484173158311)
    assert etas[3] == pytest.approx(0.0)


def test_simplex_is_fully_reducible():
    hg = _simplex_hypergraph()
    eta, reps = reducibility(hg, entropy_method="count", optimization="exact")
    assert eta == pytest.approx(1.0)
    assert reps == (3,)
    etas = layer_reducibility(hg, entropy_method="count")
    assert etas[2] == pytest.approx(1.0)
    assert etas[3] == pytest.approx(0.0)


def test_reducibility_invalid_ent_method():
    hg = _small_hypergraph()
    with pytest.raises(InvalidParameterError):
        reducibility(hg, entropy_method="invalid")


def test_reducibility_invalid_optimization():
    hg = _small_hypergraph()
    with pytest.raises(InvalidParameterError):
        reducibility(hg, optimization="invalid")


def test_partition_list_wrong_length():
    hg = _small_hypergraph()
    with pytest.raises(InvalidParameterError):
        reducibility(hg, partition=[0, 1], entropy_method="count")


def test_partition_dict_missing_node():
    hg = _small_hypergraph()
    with pytest.raises(InvalidParameterError):
        reducibility(hg, partition={0: 0, 1: 0, 2: 1}, entropy_method="count")
