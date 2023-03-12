def intersection(a, b):
    return len(a.intersection(b))


def jaccard(a, b):
    """
    Compute the Jaccard similarity between two sets.
    Parameters
    ----------
    a : set. The first set.
    b : set. The second set.

    Returns
    -------
    float. The Jaccard similarity between the two sets.
    """
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / len(a.union(b))


def jaccard_distance(a, b):
    """
    Compute the Jaccard distance between two sets.
    Parameters
    ----------
    a : set. The first set.
    b : set. The second set.

    Returns
    -------
    float. The Jaccard distance between the two sets. The distance is 1 - the similarity.
    """
    return 1 - jaccard(a, b)

