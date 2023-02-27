def intersection(a, b):
    return len(a.intersection(b))


def jaccard(a, b):
    return len(a.intersection(b)) / len(a.union(b))
