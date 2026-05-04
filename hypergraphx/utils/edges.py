def canon_edge(edge):
    edge = tuple(edge)

    if len(edge) == 2:
        first_is_group = isinstance(edge[0], (tuple, list))
        second_is_group = isinstance(edge[1], (tuple, list))
        if first_is_group and second_is_group:
            return (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if not first_is_group and not second_is_group:
            return tuple(sorted(edge))

    return tuple(sorted(edge))
