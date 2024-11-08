import numpy as np

from hypergraphx import DirectedHypergraph


def hyperedge_signature_vector(hypergraph: DirectedHypergraph, max_hyperedge_size=None):
    """
    Compute the hyperedge signature vector of a directed hypergraph. The hyperedge signature is a vector that counts the
    number of hyperedges for each combination of source and target size. The signature is computed up to a maximum hyperedge size. If the maximum hyperedge
    size is not provided, the maximum size of the hyperedges in the hypergraph is used.

    Parameters
    ----------
    hypergraph: DirectedHypergraph
        The directed hypergraph for which to compute the hyperedge signature vector.
    max_hyperedge_size: int, optional
        The maximum hyperedge size to consider in the signature. If not provided, the maximum size of the hyperedges in the hypergraph is used.

    Returns
    -------
    numpy.ndarray
        The hyperedge signature vector.

    Raises
    ------
    ValueError
        If the hypergraph is not a DirectedHypergraph.

    Examples
    --------
    >>> from hypergraphx import DirectedHypergraph
    >>> from hypergraphx.measures.directed import hyperedge_signature_vector
    >>> edges = [
    ...     ((1, 2), (3, 4)),  # Hyperedge with source size 2, target size 2
    ...     ((5, ), (6, 7, 8)),  # Hyperedge with source size 1, target size 3
    ... ]
    >>> hypergraph = DirectedHypergraph(edges)
    >>> result = hyperedge_signature_vector(hypergraph)
    # output
    array([0, 0, 1, 0, 1, 0, 0, 0, 0])
    """

    if not isinstance(hypergraph, DirectedHypergraph):
        raise ValueError("The hypergraph must be a DirectedHypergraph")

    if max_hyperedge_size is None:
        try:
            max_hyperedge_size = max(hypergraph.get_sizes())
        except ValueError:
            return np.array([])

    signature = np.zeros((max_hyperedge_size-1, max_hyperedge_size-1))

    for hyperedge in hypergraph.get_edges(size=max_hyperedge_size, up_to=True):
        source_size = len(hyperedge[0])
        target_size = len(hyperedge[1])
        signature[source_size-1, target_size-1] += 1

    signature = np.array(signature.flatten())
    return signature