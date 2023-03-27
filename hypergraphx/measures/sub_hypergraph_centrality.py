import numpy as np
from scipy import special

from hypergraphx import Hypergraph


def subhypergraph_centrality(hypergraph: Hypergraph) -> np.ndarray:
    """Compute the logarithm of the sub-hypergraph centrality, defined in
    "Complex Networks as Hypergraphs",
    Estrada & Rodríguez-Velázquez, 2005

    For every node v in the hypergraph, the sub-hypergraph centrality is given by the
    number of closed random walks starting and ending at v, each one discounted by the
    factorial of its length. This function computes the logarithm of the centrality
    defined in the reference paper.


    Parameters
    ----------
    hypergraph: the hypergraph.

    Returns
    -------
    The array of the log-sub-hypergraph centrality values for all the nodes in the
    hypergraph.
    """
    adj = hypergraph.adjacency_matrix().todense()
    eigenvals, eigenvecs = np.linalg.eigh(adj)
    return special.logsumexp(eigenvals.reshape(1, -1), b=eigenvecs ** 2, axis=1)
