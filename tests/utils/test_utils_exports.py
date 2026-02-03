import hypergraphx.utils as u


def test_utils_exports_labeling_and_metadata():
    assert isinstance(u.merge_metadata({"a": 1}, {"a": 2})["a"], list)

    enc = u.LabelEncoder().fit(["x", "y"])
    assert list(enc.transform(["x", "y"])) == [0, 1]


def test_utils_lazy_submodules():
    assert u.edges.canon_edge((2, 1)) == (1, 2)
    assert u.metadata.merge_metadata({"a": 1}, {"a": 2}) == {"a": [1, 2]}
