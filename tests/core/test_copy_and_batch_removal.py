"""Copies are independent of their original, and removing many nodes or edges at once
leaves the hypergraph exactly as removing them one at a time does."""

import random

import pytest

from hypergraphx import Hypergraph, TemporalHypergraph
from hypergraphx.exceptions import MissingEdgeError, MissingNodeError


def _random(temporal, weighted, seed):
    rng = random.Random(seed)
    edges = [tuple(rng.sample(range(30), rng.randint(2, 5))) for _ in range(150)]
    weights = [rng.randint(1, 5) for _ in edges] if weighted else None
    if temporal:
        h = TemporalHypergraph(
            edge_list=edges,
            time_list=[rng.choice([0, 2, 3, 7]) for _ in edges],
            weighted=weighted,
            weights=weights,
        )
        for t, e in list(h.get_edges())[:20]:
            h.set_edge_metadata(e, t, {"k": rng.random()})
    else:
        h = Hypergraph(edge_list=edges, weighted=weighted, weights=weights)
        for e in list(h.get_edges())[:20]:
            h.set_edge_metadata(e, {"k": rng.random()})
        e = h.get_edges()[0]
        h.set_incidence_metadata(e, e[0], {"role": "first"})
    for v in h.get_nodes():
        h.set_node_metadata(v, {"group": v % 3, "tags": [v]})
    return h


CASES = [(t, w, s) for t in (False, True) for w in (False, True) for s in range(5)]


@pytest.mark.parametrize("temporal,weighted,seed", CASES)
def test_a_copy_is_equal_and_independent(temporal, weighted, seed):
    h = _random(temporal, weighted, seed)
    c = h.copy()
    assert type(c) is type(h) and c.__dict__ == h.__dict__
    node = h.get_nodes()[0]
    c.get_node_metadata(node)["tags"].append("changed")
    c.set_node_metadata(node, {"group": "changed"})
    assert h.get_node_metadata(node) == {"group": node % 3, "tags": [node]}
    first = next(iter(h._edge_metadata))
    c._edge_metadata[first]["added"] = 1
    assert "added" not in h._edge_metadata[first]
    if temporal:
        t, e = c.get_edges()[0]
        c.remove_edge(e, t)
    else:
        c.remove_edge(c.get_edges()[0])
    assert h.num_edges() == c.num_edges() + 1


@pytest.mark.parametrize("temporal,weighted,seed", CASES)
def test_removing_many_nodes_is_removing_them_one_by_one(temporal, weighted, seed):
    h = _random(temporal, weighted, seed)
    drop = random.Random(seed).sample(h.get_nodes(), 10)
    one_by_one, at_once = h.copy(), h.copy()
    for v in drop:
        one_by_one.remove_node(v)
    at_once.remove_nodes(drop)
    assert at_once.__dict__ == one_by_one.__dict__


@pytest.mark.parametrize("temporal,weighted,seed", CASES)
def test_removing_many_edges_is_removing_them_one_by_one(temporal, weighted, seed):
    h = _random(temporal, weighted, seed)
    chosen = random.Random(seed).sample(h.get_edges(), 60)
    one_by_one, at_once = h.copy(), h.copy()
    if temporal:
        for t, e in chosen:
            one_by_one.remove_edge(e, t)
        at_once.remove_edges([e for _, e in chosen], [t for t, _ in chosen])
        packed = h.copy()
        packed.remove_edges(chosen)
        assert packed.__dict__ == one_by_one.__dict__
    else:
        for e in chosen:
            one_by_one.remove_edge(e)
        at_once.remove_edges(chosen)
    assert at_once.__dict__ == one_by_one.__dict__


def test_missing_nodes_and_edges_raise_before_anything_is_removed():
    h = _random(False, False, 0)
    for nodes in ([999], [h.get_nodes()[0], 999]):
        before = h.copy()
        with pytest.raises(MissingNodeError):
            h.remove_nodes(nodes)
        assert h.__dict__ == before.__dict__
    edge = h.get_edges()[0]
    for edges in ([(98, 99)], [edge, edge]):
        before = h.copy()
        with pytest.raises(MissingEdgeError):
            h.remove_edges(edges)
        assert h.__dict__ == before.__dict__
