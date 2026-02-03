def test_imports_smoke():
    import hypergraphx  # noqa: F401
    from hypergraphx import (
        DirectedHypergraph,
        Hypergraph,
        MultiplexHypergraph,
        TemporalHypergraph,
    )  # noqa: F401,E501

    import hypergraphx.readwrite  # noqa: F401
    import hypergraphx.linalg  # noqa: F401
    import hypergraphx.viz  # noqa: F401
