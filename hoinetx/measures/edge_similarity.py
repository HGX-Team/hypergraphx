def intersection(a, b):
    return len(a.intersection(b))


def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / len(a.union(b))


def jaccard_distance(a, b):
    return 1 - jaccard(a, b)

