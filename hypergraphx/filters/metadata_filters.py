def filter_hypergraph(
    hypergraph,
    node_criteria=None,
    edge_criteria=None,
    mode="keep",
    keep_edges=False,
    *,
    inplace: bool = True,
    node_criteria_mode: str = "all",
    edge_criteria_mode: str = "all",
):
    """
    Filters nodes and edges of a hypergraph based on metadata attributes and allowed values.

    Parameters
    ----------
    hypergraph : object
        The hypergraph instance to filter. Must have `_node_metadata` and `_edge_metadata` attributes,
        along with `remove_node` and `remove_edge` methods.
    node_criteria : dict, optional
        A dictionary specifying metadata attribute keys and allowed values for nodes.
        Example: {"type": ["person", "animal"]}.
    edge_criteria : dict, optional
        A dictionary specifying metadata attribute keys and allowed values for edges.
        Example: {"relationship": ["friendship", "collaboration"]}.
    mode : str, optional
        Either "keep" to retain only matching nodes and edges, or "remove" to discard them. Default is "keep".
    keep_edges : bool, optional
        If False (default), edges involving removed nodes are fully removed.
        If True, edges are kept but the removed nodes are excluded from the edges.
    node_criteria_mode : {"all", "any"}, optional
        How to combine multiple node criteria. Default is "all".
    edge_criteria_mode : {"all", "any"}, optional
        How to combine multiple edge criteria. Default is "all".

    Returns
    -------
    object | None
        If `inplace=True` (default), the hypergraph is modified in place and None is returned.
        If `inplace=False`, a filtered copy is returned.

    Raises
    ------
    ValueError
        If the mode is not "keep" or "remove".
    """

    if mode not in {"keep", "remove"}:
        raise ValueError('Mode must be either "keep" or "remove".')
    if node_criteria_mode not in {"all", "any"}:
        raise ValueError('node_criteria_mode must be either "all" or "any".')
    if edge_criteria_mode not in {"all", "any"}:
        raise ValueError('edge_criteria_mode must be either "all" or "any".')

    if not inplace:
        hypergraph = hypergraph.copy()

    def _matches_one(metadata, attr, predicate_or_values):
        value = metadata.get(attr, None)
        if callable(predicate_or_values):
            return bool(predicate_or_values(value))
        # Allow scalar or iterable of allowed values.
        if isinstance(predicate_or_values, (list, tuple, set, frozenset)):
            return value in predicate_or_values
        return value == predicate_or_values

    def matches_criteria(metadata, criteria, criteria_mode: str):
        if not criteria:
            return True
        results = [
            _matches_one(metadata, attr, predicate_or_values)
            for attr, predicate_or_values in criteria.items()
        ]
        return all(results) if criteria_mode == "all" else any(results)

    if node_criteria is not None:
        nodes_to_process = []
        for node, node_metadata in hypergraph.get_nodes(metadata=True).items():
            matches = matches_criteria(node_metadata, node_criteria, node_criteria_mode)
            if (mode == "keep" and not matches) or (mode == "remove" and matches):
                nodes_to_process.append(node)

        for node in nodes_to_process:
            hypergraph.remove_node(node, keep_edges=keep_edges)

    if edge_criteria is not None:
        edges_to_process = []
        for edge, edge_metadata in hypergraph.get_edges(metadata=True).items():
            matches = matches_criteria(edge_metadata, edge_criteria, edge_criteria_mode)
            if (mode == "keep" and not matches) or (mode == "remove" and matches):
                edges_to_process.append(edge)

        for edge in edges_to_process:
            hypergraph.remove_edge(edge)

    if not inplace:
        return hypergraph
