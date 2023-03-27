def intersection(a, b):
    """
    Computes the intersection between two sets.

    Parameters
    ----------
    a: set
        The first set.
    b: set
        The second set.

    Returns
    -------
    int
        The size of the intersection between the two sets.

    Example
    -------
    >>> intersection({1, 2, 3}, {2, 3, 4})
    2
    """
    return len(a.intersection(b))


def jaccard_similarity(a, b):
    """
    Computes the Jaccard similarity between two sets.

    Parameters
    ----------
    a : set
        The first set.
    b : set
        The second set.

    Returns
    -------
    float
        The Jaccard similarity between the two sets.

    See Also
    --------
    jaccard_distance

    Example
    -------
    >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})
    0.5
    """
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / len(a.union(b))


def jaccard_distance(a, b):
    """
    Compute the Jaccard distance between two sets.

    Parameters
    ----------
    a : set
        The first set.
    b : set
        The second set.

    Returns
    -------
    float
        The Jaccard distance between the two sets. The distance is 1 - the similarity.

    See Also
    --------
    jaccard_similarity

    Example
    -------
    >>> jaccard_distance({1, 2, 3}, {2, 3, 4})
    0.5
    """
    return 1 - jaccard_similarity(a, b)

