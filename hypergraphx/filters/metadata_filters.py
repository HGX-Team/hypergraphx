def filter_hypergraph(
    hypergraph, node_criteria=None, edge_criteria=None, mode="keep", keep_edges=False
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

    Returns
    -------
    None
        The hypergraph is modified in place.

    Raises
    ------
    ValueError
        If the mode is not "keep" or "remove".
    """

    if mode not in {"keep", "remove"}:
        raise ValueError('Mode must be either "keep" or "remove".')

    def matches_criteria(metadata, criteria):
        return all(metadata.get(attr) in values for attr, values in criteria.items())

    if node_criteria is not None:
        nodes_to_process = []
        for node, node_metadata in hypergraph.get_nodes(metadata=True).items():
            matches = matches_criteria(node_metadata, node_criteria)
            if (mode == "keep" and not matches) or (mode == "remove" and matches):
                nodes_to_process.append(node)

        for node in nodes_to_process:
            hypergraph.remove_node(node, keep_edges=keep_edges)

    if edge_criteria is not None:
        edges_to_process = []
        for edge, edge_metadata in hypergraph.get_edges(metadata=True).items():
            matches = matches_criteria(edge_metadata, edge_criteria)
            if (mode == "keep" and not matches) or (mode == "remove" and matches):
                edges_to_process.append(edge)

        for edge in edges_to_process:
            hypergraph.remove_edge(edge)
