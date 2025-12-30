import logging
import networkx as nx
import numpy as np
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import fcluster, linkage

from hypergraphx import Hypergraph
from hypergraphx.measures.edge_similarity import jaccard_distance as jaccard
from hypergraphx.readwrite.io_pickle import load_pickle, save_pickle


def hyperlink_communities(
    H: Hypergraph, load_distances=None, save_distances=None
) -> np.ndarray:
    """
    Computes the dendrogram of the given hypergraph

    Parameters
    ----------
    H : Hypergraph
        The hypergraph of interest
    load_distances : str
        The path to load the distances from
    save_distances : str
        The path to save the distances to

    Returns
    -------
    np.ndarray
        The dendrogram of the given hypergraph

    """
    logger = logging.getLogger(__name__)
    logger.info("Hypergraph info - nodes: %s edges: %s", H.num_nodes(), H.num_edges())
    lcc = H.largest_component()
    H = H.subhypergraph(lcc)
    h = H.get_edges()
    logger.info(
        "Subhypergraph info - nodes: %s edges: %s", H.num_nodes(), H.num_edges()
    )

    adj = {}
    edge_to_id = {}

    cont = 0
    for e in h:
        e = tuple(sorted(e))
        edge_to_id[e] = cont
        for n in e:
            if n not in adj:
                adj[n] = []
            adj[n].append(e)
        cont += 1

    logger.info("Computing distances")

    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(h))])

    try:
        X = load_pickle("{}.hlcd".format(load_distances))
    except FileNotFoundError:
        vis = {}
        c = 0
        for n in adj:
            logger.info("Done %s of %s", c, len(adj))
            for i in range(len(adj[n]) - 1):
                for j in range(i + 1, len(adj[n])):
                    k = tuple(sorted((edge_to_id[adj[n][i]], edge_to_id[adj[n][j]])))
                    e_i = set(adj[n][i])
                    e_j = set(adj[n][j])
                    if k not in vis:
                        w = jaccard(e_i, e_j)
                        if w > 0:
                            G.add_edge(
                                edge_to_id[adj[n][i]], edge_to_id[adj[n][j]], weight=w
                            )
                        vis[k] = True
            c += 1

        X = nx.to_numpy_array(G, weight="weight", nonedge=1.0)
        save_pickle(X, "{}.hlcd".format(save_distances))

    logger.info("dist computed")

    np.fill_diagonal(X, 0.0)
    dist_matrix = ssd.squareform(X)
    dendrogram = linkage(dist_matrix, method="average")
    return dendrogram


def _cut_dendrogram(dendrogram, cut_height):
    cut = fcluster(dendrogram, t=cut_height, criterion="distance")
    return cut


def _edge_label2node_labels(h, labels):
    nodes = {}
    for i in range(len(labels)):
        label_arco = labels[i]
        for nodo in h[i]:
            if nodo not in nodes:
                nodes[nodo] = []
                nodes[nodo].append(label_arco)
            else:
                nodes[nodo].append(label_arco)
    return nodes


def get_num_hyperlink_communties(dendrogram: np.ndarray, cut_height: float) -> int:
    """
    Returns the number of communities in the dendrogram at the cut height

    Parameters
    ----------
    dendrogram : np.ndarray

    cut_height : float
        The cut height
    Returns
    -------
    int
        The number of communities
    """
    cut = _cut_dendrogram(dendrogram, cut_height)
    return len(set(cut))


def overlapping_communities(
    H: Hypergraph, dendrogram: np.ndarray, cut_height: float
) -> dict:
    """
    Returns the overlapping communities in the dendrogram at the given cut height

    Parameters
    ----------
    H : Hypergraph
        The hypergraph of interest
    dendrogram : np.ndarray
        The dendrogram of the given hypergraph
    cut_height
        The cut height
    Returns
    -------
    dict
        The overlapping communities
    """
    cut = _cut_dendrogram(dendrogram, cut_height)
    h = H.get_edges()
    labels = _edge_label2node_labels(h, cut)
    return labels
