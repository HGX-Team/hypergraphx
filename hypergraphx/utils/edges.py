def canon_edge(edge):
    edge = tuple(edge)

    if len(edge) == 2:
        if isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
            return (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if not isinstance(edge[0], tuple) and not isinstance(edge[1], tuple):
            return tuple(sorted(edge))

    return tuple(sorted(edge))
