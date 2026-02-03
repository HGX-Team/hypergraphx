import hypergraphx as hgx
import logging
import pickle as pk
from hypergraphx.representations.projections import clique_projection
import numpy as np
import pandas as pd
import networkx as nx
from hypergraphx.readwrite import load_hypergraph
import time
import warnings

logger = logging.getLogger(__name__)


def _log(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    logger.info(message)


# ========= MAIN FUNCTIONS ======== #


def calc_HO_shortest_paths(
    Hbase: hgx.Hypergraph, option="min", root=None, verbose=False
):
    """
    Calculate shortest paths in a hypergraph and analyze the orders of hyperedges along these paths.
    Parameters
    ----------
    Hbase : hgx.Hypergraph
        The input hypergraph.
    option : str
        The method to use for determining the order of hyperedges along the paths.
        Options are 'min', 'max', or 'mean'.
    root : str, optional
        The directory path to save the results. If None, results are not saved.

    Returns
    -------
    tuple
        - SPL_ho (pd.DataFrame): DataFrame of shortest path lengths in the full hypergraph.
        - SPL_dy (pd.DataFrame): DataFrame of shortest path lengths in the dyadic subhypergraph.
        - SPL_onlyho (pd.DataFrame): DataFrame of shortest path lengths in the higher-order only subhypergraph.
        - avg_ord (pd.DataFrame): DataFrame of average orders of hyperedges along the shortest paths.

    """

    # check indexation of nodes
    assert (
        min(list(Hbase.get_nodes())) == 0
    ), f"the network should have nodes starting from 0!, but instead they start from {min(list(H__base.get_nodes()))} \nHas the stattic graph for {ds} been correctly calculated from the TN?"

    start = time.time()

    if root is None:
        _log("No root directory provided, will not save any intermediate results")
    else:
        warnings.warn(
            "calc_HO_shortest_paths(..., root=...) performs file I/O. "
            "Prefer root=None for library usage; use io_* helpers for file workflows.",
            DeprecationWarning,
            stacklevel=2,
        )

    assert option in [
        "min",
        "max",
        "mean",
    ], "option must be one of ['min', 'max', 'mean']"
    if verbose:
        _log(f"Hyperedge selection strategy: {option.upper()}")

    Hbase = HO_convert_node_labels_to_integers(Hbase)

    # Get the dyadic and only higher-order subhypergraphs
    H_dyads = Hbase.subhypergraph_by_orders(sizes=[2], keep_nodes=True)
    H_onlyHO = Hbase.subhypergraph_by_orders(
        sizes=range(3, Hbase.max_size() + 1), keep_nodes=True
    )

    # check that it's even worth calculating doing this analysis
    assert not (
        sorted(clique_projection(Hbase, keep_isolated=True))
        == sorted(clique_projection(H_dyads, keep_isolated=True).edges())
    ), "The projected graphs should be different from the original graph if we're planning to investigate higher orders"

    _log("A. Calculating shortest paths")
    # For dyads
    shortest_paths_lengths_dy, SPL_dy = calc_ho_shortest_paths(H_dyads)
    # For only higher orders
    shortest_paths_lengths_onlyho, SPL_onlyho = calc_ho_shortest_paths(H_onlyHO)
    # For full hypergraph
    shortest_paths_ho, SPL_ho = calc_ho_shortest_paths(Hbase)

    if root is not None:
        fname = f"{root}/" + "{f}.pck"
        pk.dump(SPL_ho, open(fname.format(f="spl_ho"), "wb"))
        pk.dump(SPL_onlyho, open(fname.format(f="spl_onlyho"), "wb"))
        pk.dump(SPL_dy, open(fname.format(f="spl_dy"), "wb"))
        pk.dump(shortest_paths_ho, open(fname.format(f="all_shortest_paths_ho"), "wb"))

    _log("B. calculating orders and avg orders and redundancies of ho paths")
    shortest_paths_ho_dict_enriched = calc_sizes_redundancies_of_shortest_paths(
        shortest_paths_ho=shortest_paths_ho, Hbase=Hbase, option=option, root=root
    )

    _log("C. extracting avg orders of ho paths")
    nodes = Hbase.get_nodes()

    # Fill a matrix with nan values
    avg_ord = np.empty((len(nodes), len(nodes)))
    avg_ord.fill(np.nan)
    for u, dicto in shortest_paths_ho_dict_enriched.items():
        for v, spl in dicto.items():
            # skip self-paths
            if u != v and len(spl["sizes"]) >= 1:
                avg_ord[u, v] = spl["avg_size"]

    avg_ord = pd.DataFrame(data=avg_ord, index=sorted(nodes), columns=sorted(nodes))

    end = time.time()
    if verbose:
        _log(f"Time taken for shortest paths and avg orders = {end-start:.2f}s \n")

    # save the average orders
    if root is not None:
        fname = f"{root}/" + "{f}.pck"
        pk.dump(avg_ord, open(fname.format(f=f"avg_ord_{option}"), "wb"))

    return SPL_ho, SPL_dy, SPL_onlyho, avg_ord, shortest_paths_ho_dict_enriched


def calc_HO_shortest_paths_from_file(fnameho, option="min", root=None, verbose=False):
    warnings.warn(
        "calc_HO_shortest_paths_from_file(...) is deprecated; use "
        "io_calc_HO_shortest_paths_from_file(...).",
        DeprecationWarning,
        stacklevel=2,
    )
    return io_calc_HO_shortest_paths_from_file(
        fnameho, option=option, root=root, verbose=verbose
    )


def io_calc_HO_shortest_paths_from_file(
    fnameho, option="min", root=None, verbose=False
):
    """
    Calculate shortest paths in a hypergraph and analyze the orders of hyperedges along these paths.

    Alternative to calc_HO_shortest_paths() where instead of providing a hgx.Hypergraph object,
    the hypergraph is loaded from a file.

    """

    # Load the hypergraph
    Hbase = load_hypergraph(fnameho)

    (
        SPL_ho,
        SPL_dy,
        SPL_onlyho,
        avg_ord,
        shortest_paths_ho_dict_enriched,
    ) = calc_HO_shortest_paths(Hbase=Hbase, option=option, root=root, verbose=verbose)

    return SPL_ho, SPL_dy, SPL_onlyho, avg_ord, shortest_paths_ho_dict_enriched


# ========= HELPER FUNCTIONS ======== #


def dict_to_df(dict_sps, nodes):
    """
    Converts a NetworkX shortest path dictionary to a pandas DataFrame of shortest path lengths.

    Parameters
    ----------
    dict_sps : dict
        Dictionary of shortest paths generated from NetworkX, where keys are source nodes and values are
        dictionaries mapping target nodes to shortest path lengths.
    nodes : list
        List of node identifiers to be used as DataFrame indices and columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame of shortest path lengths, indexed and columned by node IDs. Paths from a node to itself are set to NaN.
    """
    nodes = sorted(nodes)  # sort to ensure consistent ordering for indices

    SPL_ho = np.empty((len(nodes), len(nodes)))
    SPL_ho.fill(np.nan)

    for u, dicto in dict_sps.items():
        for v, spl in dicto.items():
            if u != v:  # keep self-paths as nan
                SPL_ho[u, v] = spl

    # convert to dataframe
    SPL_ho = pd.DataFrame(data=SPL_ho, index=sorted(nodes), columns=sorted(nodes))

    return SPL_ho


def HO_convert_node_labels_to_integers(H: hgx.Hypergraph) -> hgx.Hypergraph:
    def relabel(edges: list, relabeling: dict):
        """
        Relabel the vertices of a hypergraph according to a given relabeling

        Parameters
        ----------
        edges : list
            Edges of the hypergraph
        relabeling : dict
            Relabeling

        Returns
        -------
        list
            Edges of the hypergraph with the vertices relabeled

        Notes
        -----
        The relabeling is a dictionary that maps the old labels to the new labels
        """
        res = []
        for edge in edges:
            new_edge = []
            for v in edge:
                new_edge.append(relabeling[v])
            res.append(tuple(sorted(new_edge)))
        return sorted(res)

    oldelist = H.get_edges()

    mapping = {old: new for (old, new) in zip(H.get_nodes(), range(H.num_nodes()))}

    newelist = relabel(edges=oldelist, relabeling=mapping)

    Hnew = hgx.Hypergraph(newelist)
    return Hnew


# ========= AUXILIARY FUNCTIONS ======== #


def calc_ho_shortest_paths(H: hgx.Hypergraph):
    """
    Calculates higher-order shortest paths in a hypergraph using clique projection.
    Args:
        H (hgx.Hypergraph): The input hypergraph.
    Returns:
        tuple:
            - shortest_paths_ho (dict): Dictionary of shortest paths between all pairs of nodes in the projected graph.
            - SPL_ho (pd.DataFrame): DataFrame representation of shortest path lengths between nodes.

    """

    G = clique_projection(H, keep_isolated=True)
    shortest_paths_ho = dict(nx.all_pairs_shortest_path(G))
    SPL_ho = dict_to_df(dict_sps=shortest_paths_ho, nodes=H.get_nodes())

    return shortest_paths_ho, SPL_ho


def calc_sizes_redundancies_of_shortest_paths(
    shortest_paths_ho, Hbase, option, root=None
):
    """
    Calculate the orders of hyperedges along the shortest paths in a hypergraph.

    Parameters
    ----------
    shortest_paths_ho : dict
        A dictionary containing the shortest paths between all pairs of nodes in the hypergraph.
        Format: {node1: {node2: [path_nodes]}, ...}
    Hbase : hgx.Hypergraph
        The original hypergraph from which the shortest paths were derived.
    option : str
        The method to use for determining the order of hyperedges along the paths.
        Options are 'min', 'max', or 'mean'.
    root : str, optional
        The directory path to save the results. If None, results are not saved.

    Returns
    -------
    dict
        Updated shortest_paths_ho dictionary with added information about hyperedge orders and redundancies.
        Format: {node1: {
                    node2: {
                        'sp': [path_nodes],
                        'sizes': [orders],
                        'redundancies': [redundancies]}
                        },
                    ...
                }

    """
    if root is not None:
        warnings.warn(
            "calc_sizes_redundancies_of_shortest_paths(..., root=...) performs file I/O. "
            "Prefer root=None for library usage.",
            DeprecationWarning,
            stacklevel=2,
        )

    for ctr, (node_i, dicto) in enumerate(shortest_paths_ho.items()):
        _log(f"{ctr}\t/{len(shortest_paths_ho)}", end="\r")
        for node_j, shortest_path in dicto.items():
            path_sizes = np.zeros(len(shortest_path) - 1)
            redunancies = np.zeros(len(shortest_path) - 1)

            if node_i == node_j:
                # skip self-paths
                continue
            else:
                for ctr, (k, l) in enumerate(
                    zip(shortest_path[:-1], shortest_path[1:])
                ):
                    # find all edges containing both nodes i and j
                    incident_edges = set(Hbase.get_incident_edges(node=k)).intersection(
                        set(Hbase.get_incident_edges(node=l))
                    )
                    # for each step, if there are multiple edges containing these two nodes, use strategy [option] over possible edges
                    match option:
                        case "min":
                            size_nodepair = len(min(incident_edges, key=len))
                        case "max":
                            size_nodepair = len(max(incident_edges, key=len))
                        case "mean":
                            size_nodepair = np.mean([len(e) for e in incident_edges])
                        case _:
                            _log("No option selected")
                            raise ValueError

                    assert size_nodepair != 1
                    path_sizes[ctr] = size_nodepair
                    redunancies[ctr] = (
                        len(incident_edges) - 1
                    )  # no.incident edges = 1 + no.redundant edges

            # expand the shortest_path_ho dict to include path orders and redundancies
            shortest_paths_ho[node_i][node_j] = {
                "sp": shortest_paths_ho[node_i][node_j],
                "sizes": path_sizes,
                "redundancies": redunancies,
                "avg_size": np.array(path_sizes).mean(),
            }

    if root is not None:
        fname = f"{root}/" + "{f}.pck"
        fname = fname.format(f=f"all_shortest_paths_ho_{option}")
        pk.dump(shortest_paths_ho, open(fname, "wb"))

    return shortest_paths_ho


def calc_prop_true_dyad_paths_per_spl(SPL_ho, avg_ord):
    temp = pd.DataFrame(
        data=[SPL_ho.ravel(), avg_ord.ravel()], index=["spl", "avg_ord"]
    ).T
    temp = temp[temp["spl"] != 0]
    temp["is_dyad"] = temp["avg_ord"] == 2
    temp = temp.drop(columns=["avg_ord"]).groupby("spl").value_counts(normalize=True)
    temp = temp.unstack()
    temp.columns = ["prop_False", "prop_True"]
    temp = temp.reset_index()
    return temp


def calc_prop_of_each_path_is_dyad(shortest_paths_ho, SPL_ho):
    prop_dyad_path = np.empty((len(SPL_ho), len(SPL_ho)))
    prop_dyad_path.fill(np.nan)

    for person1, subdict in shortest_paths_ho.items():
        for person2, sp_avg_ord_dict in subdict.items():
            if person1 == person2:
                continue
            sp = sp_avg_ord_dict["sp"]
            sizes = sp_avg_ord_dict["sizes"]

            _pd = sum(sizes == 2) / len(sizes)

            prop_dyad_path[person1][person2] = _pd

    temp = pd.DataFrame(
        data=[SPL_ho.to_numpy().ravel(), prop_dyad_path.ravel()], index=["spl", "pd"]
    ).T
    temp = temp[temp["spl"] != 0]
    temp = temp.groupby("spl").mean()
    temp = temp.reset_index()

    return temp
