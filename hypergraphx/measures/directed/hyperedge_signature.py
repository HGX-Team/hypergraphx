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
    """

    if not isinstance(hypergraph, DirectedHypergraph):
        raise ValueError("The hypergraph must be a DirectedHypergraph")

    if max_hyperedge_size is None:
        max_hyperedge_size = hypergraph.max_size()

    signature = np.zeros(max_hyperedge_size, max_hyperedge_size)

    for hyperedge in hypergraph.get_edges(size=max_hyperedge_size, up_to=True):
        source_size = len(hyperedge[0])
        target_size = len(hyperedge[1])
        signature[source_size, target_size] += 1

    signature = np.array(signature.flatten())
    return signature