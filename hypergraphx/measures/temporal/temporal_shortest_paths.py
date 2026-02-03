import hypergraphx as hgx
import json
import logging
from datetime import datetime
from hypergraphx.representations.projections import clique_projection
from hypergraphx.utils.labeling import relabel_edges_with_mapping

import os
import numpy as np
import pickle as pk
import networkx as nx
import pandas as pd
from copy import deepcopy
import warnings

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover

    def tqdm(it, **kwargs):
        return it


try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover
    plt = None

logger = logging.getLogger(__name__)


def _log(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    logger.info(message)


def relabel(edges, mapping):
    """Relabel edges using a dictionary mapping old labels to new labels."""
    return relabel_edges_with_mapping(edges=edges, mapping=mapping)


def get_ds_windowsize(nsamples):
    if nsamples < 500:
        return nsamples
    else:
        return 300


def supra_adj(temporal_network, subtimes, unique_individuals, dataset):
    """
    Create a supra-adjacency representation of a temporal network

    Parameters
    ----------
    subnetwork : dict
        A dictionary of {time: hypergraph or graph} representing the temporal network
    subtimes : list
        A list of times representing the times to consider in the temporal network
    unique_individuals : set
        A set of unique individuals in the temporal network
    dataset : str
        The name of the dataset, used to determine time difference calculation

    Returns
    -------
    G : nx.DiGraph
        A directed graph representing the supra-adjacency representation of the temporal network


    """

    G = nx.DiGraph()
    for t in subtimes:
        nodes = [
            (t, n)
            for n in (
                temporal_network[t].get_nodes()
                if isinstance(temporal_network[t], hgx.Hypergraph)
                else temporal_network[t].nodes()
            )
        ]
        G.add_nodes_from(nodes)

    # for successively each subnetwork
    for i, t in enumerate(subtimes):
        subnet = temporal_network[t]
        # for each node
        NODES = (
            subnet.get_nodes() if isinstance(subnet, hgx.Hypergraph) else subnet.nodes()
        )
        for n in NODES:
            # find all next active agents and link directly
            j = i + 1
            found = False
            while j < len(subtimes) and not found:
                t_j = subtimes[j]
                # print(i, j, n)
                active_in = (
                    (n in set(temporal_network[subtimes[j]].get_nodes()))
                    if isinstance(temporal_network[subtimes[j]], hgx.Hypergraph)
                    else (n in set(temporal_network[subtimes[j]].nodes()))
                )
                if active_in:
                    found = True
                    if "FnF" in dataset:
                        # temp1 = datetime.strptime(t_j[:-1], "%Y-%m-%d_%H")
                        # temp2 = datetime.strptime(t[:-1], "%Y-%m-%d_%H")
                        # hours_passed = (temp1 - temp2).total_seconds()/3600
                        hours_passed = t_j - t
                    else:
                        hours_passed = t_j - t

                    assert hours_passed > 0, f"{t_j} - {t} = {t_j-t}"
                    G.add_edge((t, n), (t_j, n), weight=hours_passed)
                j += 1

    # then go back and do cross-diag coupling
    for i, t in enumerate(subtimes):
        subnet = temporal_network[t]
        # for each edge, find cross-diagonal
        if isinstance(subnet, hgx.Hypergraph):
            edges = subnet.get_edges()
            raise NotImplementedError("Currently not yet extended to hypergraphs")
        elif isinstance(subnet, nx.classes.graph.Graph):
            edges = subnet.edges()
            edges_to_add = []
            for (
                e
            ) in (
                edges
            ):  # for each edge in the original snapshot, see where these nodes are active for the first time again
                n1 = (t, e[0])
                n2 = (t, e[1])
                # for each node, find cross-diagonal neighbour
                for nbr in nx.neighbors(G, n1):
                    hours_passed = nbr[0] - n2[0]

                    assert hours_passed > 0
                    edges_to_add.append((n2, nbr, {"weight": hours_passed}))

                for nbr in nx.neighbors(G, n2):
                    hours_passed = nbr[0] - n1[0]
                    assert hours_passed > 0
                    edges_to_add.append((n1, nbr, {"weight": hours_passed}))
            G.add_edges_from(
                edges_to_add
            )  # don't add too soon otherwise double-count during second adding-in
        elif isinstance(subnet, hgx.Hypergraph):
            edges = list(tuple(el) for el in subnet.get_edges())
            raise NotImplementedError("Currently not yet extended to hypergraphs")
        else:
            raise TypeError(
                f"A graph of type {type(subnet)} is not supported for SA-embeddings"
            )

    # Now add all nodes in each layer that are singletons, ie persons that don't participate in any interactions at each time
    for t in subtimes:
        not_present_nodes_t = set(temporal_network[t].nodes())
        not_present_nodes_t = set.difference(
            *[unique_individuals, not_present_nodes_t]
        )  # which persons exists, but didn't interact here?
        nodes = [(t, n) for n in list(not_present_nodes_t)]
        # numnodes+=len(nodes)
        G.add_nodes_from(nodes)

    return G


def calc_size_of_single_path(hyperpath, temporal_network, dataset, option):
    """
    Calculate the size, durations and redundancies of a single path in a temporal network

    Parameters
    ----------
    hyperpath : list
        A list of (time, node) tuples representing the path
    temporal_network : dict
        A dictionary of {time: hypergraph or graph} representing the temporal network
    dataset : str
        The name of the dataset, used to determine time difference calculation
    option : str
        The option to use when calculating the size of the path. Must be one of ['min', 'max', 'mean']

    Returns
    -------
    sizes : list
        A list of sizes of each step in the path
    timesteps : list
        A list of time differences between each step in the path
    redundancies : list
        A list of number of redundant edges at each step in the path

    """

    assert option in [
        "min",
        "max",
        "mean",
    ], "option must be one of ['min', 'max', 'mean']"

    sizes = []
    redundancies = []
    timesteps, _ = zip(*hyperpath)

    for (t1, u), (t2, v) in zip(hyperpath[:-1], hyperpath[1:]):
        # print("pair", u, t1, v, t2)
        if (
            (u == v)
            and (u not in temporal_network[t1].get_nodes())
            and (v not in temporal_network[t1].get_nodes())
        ):
            # print("Carrying over information across same individual")
            _size = 2

        else:
            if t1 == t2:
                _log("Same time. Neighbour")

            incident_edges = set(
                temporal_network[t1].get_incident_edges(node=u)
            ).intersection(set(temporal_network[t1].get_incident_edges(node=v)))

            # for each step, if there are multiple edges containing these two nodes, take length of smallest/max/aver edge over the different options
            temp = [len(el) for el in incident_edges]

            match option:
                case "min":
                    _size = min(temp)
                case "max":
                    _size = max(temp)
                case "mean":
                    _size = np.mean(temp)

        sizes.append(_size)

        redundancies.append(len(incident_edges) - 1)

    timesteps = np.diff(timesteps)

    return sizes, timesteps, redundancies


def P4_calc_shortest_fastest_paths(
    start_node_labels, static_graph_embedded, verbose=False
):
    """
    Calculate shortest paths using Dijkstra's algorithm for both fastest and shortest paths
    Shortest use an unweighted graph, fastest use a weighted graph with weights as time differences

    Parameters
    ----------
    start_node_labels : list
        A list of node labels to start the paths from
    gstatic : nx.DiGraph
        A directed graph representing the supra-adjacency representation of the temporal network
    verbose : bool
        Whether to print progress

    Returns
    -------
    fdict : dict
        A dictionary of {start_node: (lengths, paths)} for fastest paths
    sdict : dict
        A dictionary of {start_node: (lengths, paths)} for shortest paths


    """

    if verbose:
        _log("\tfastest")
    fdict = dict()
    weight = "weight"
    for ctr, n in enumerate(
        set(start_node_labels).intersection(set(static_graph_embedded.nodes))
    ):
        _log(f"Node {ctr}/{len(start_node_labels)}", end="\r")
        _t = nx.single_source_dijkstra(static_graph_embedded, source=n, weight=weight)
        fdict[n] = _t
    if verbose:
        _log("")

    if verbose:
        _log("\tshortest")
    sdict = dict()
    weight = "None"
    for ctr, n in enumerate(
        set(start_node_labels).intersection(set(static_graph_embedded.nodes))
    ):
        if verbose:
            _log(f"Node {ctr}/{len(start_node_labels)}", end="\r")
        _t = nx.single_source_dijkstra(static_graph_embedded, source=n, weight=weight)
        sdict[n] = _t
    if verbose:
        _log("")

    return fdict, sdict


def P5_calc_best_paths(
    start_node_labels, temp, sdict, fdict, integer_lbl_to_static_node_map
):
    """
    Calculate best paths by comparing fastest and shortest paths
    SQ_ARR is of the form:
    {person1: {person2: [(n1, n2, lenhoS, lenhoF, pathhoS, pathhoF), ...]}}
    where n1, n2 are (t, node) tuples
    n1 is the start node, n2 is the end node
    and lenhoS, lenhoF are the lengths of the shortest and fastest paths respectively
    and pathhoS, pathhoF are the shortest and fastest paths respectively

    Parameters
    ----------
    start_node_labels : list
        A list of node labels to start the paths from
    temp : pd.DataFrame
        A dataframe with columns ['OG-node', 'new-node'] mapping original node labels to new node labels
    sdict : dict
        A dictionary of {start_node: (lengths, paths)} for shortest paths
    fdict : dict
        A dictionary of {start_node: (lengths, paths)} for fastest paths
    integer_lbl_to_static_node_map : dict
        A dictionary mapping integer node labels to (time, node) tuples

    Returns
    -------
    SHORTEST_PATH_DATA : dict
        A dictionary of {person1: {person2: [(n1, n2, lenhoS, lenhoF, pathhoS, pathhoF), ...]}} representing the best paths
        where n1, n2 are (t, node) tuples
        n1 is the start node, n2 is the end node
        and lenhoS, lenhoF are the lengths of the shortest and fastest paths respectively
        and pathhoS, pathhoF are the shortest and fastest paths respectively


    """

    SHORTEST_PATH_DATA = dict()
    ctr = 0
    bigctr = 0
    for u, subdict in tqdm(sdict.items()):
        lengths, paths = subdict

        assert (
            u in start_node_labels
        ), "for the single_source_dijkstra algorithm, the nodes in the resulting dict MUST only be start_nodes"

        person1 = int(
            temp[temp["new-node"] == u]["OG-node"].values[0]
        )  # assert len(temp[temp['new-node']==u]
        for v, lenhoS in zip(lengths.keys(), lengths.values()):
            pathhoS = paths[v]
            person2 = int(
                temp[temp["new-node"] == v]["OG-node"].values[0]
            )  # assert len(temp[temp['new-node']==v]['OG-node'].values.tolist())==1

            if person1 == person2 or u == v:
                continue
            else:
                bigctr += 1

            lenhoF = fdict[u][0][v]
            pathhoF = fdict[u][1][v]

            assert len(pathhoF) >= len(
                pathhoS
            ), "the shortest path must be topologically shorter than the fastest path"

            if pathhoS != pathhoF:
                # print(f"S!=F for HO: {pathhoS} vs. {pathhoF}")
                ctr += 1

            # add to SHORTEST_PATH_DATA
            data = (
                integer_lbl_to_static_node_map[u],
                integer_lbl_to_static_node_map[v],
                lenhoS,
                lenhoF,
                pathhoS,
                pathhoF,
            )
            if person1 not in SHORTEST_PATH_DATA:
                SHORTEST_PATH_DATA[person1] = {person2: [data]}
            elif person2 not in SHORTEST_PATH_DATA[person1]:
                SHORTEST_PATH_DATA[person1][person2] = [data]
            else:
                SHORTEST_PATH_DATA[person1][person2].append(data)

    return SHORTEST_PATH_DATA


def P6_calc_avg_orders(
    SHORTEST_PATH_DATA, integer_lbl_to_static_node_map, TS, dataset, option
):
    """
    SQ_ARR is of the form:
    {person1: {person2: [(n1, n2, lenhoS, lenhoF, pathhoS, pathhoF), ...]}}
    where n1, n2 are (t, node) tuples

    SQ_ARR_NEW is of the form:
    {person1: {person2: {'F': (n1, n2, tdiffF, lenhoF, pathhoF, sizesf, redundanciesF),
                        'S': (n1, n2, tdiffS, lenhoS, pathhoS, sizess, redundanciesS),
                        'equal': bool (pathhoS == pathhoF)
                       }
               }
    }
    where sizesf, sizess are lists of orders of each step in the path
    and redundanciesF, redundanciesS are lists of number of redundant edges at each step in the path
    and tdiffF, tdiffS are the total time taken for the fastest/shortest path respectively

    'F' refers to fastest
    'S' refers to shortest
    """

    SHORTEST_PATH_DATA_enriched = dict()
    for person1 in tqdm(SHORTEST_PATH_DATA.keys()):
        SHORTEST_PATH_DATA_enriched[person1] = dict()
        for person2 in SHORTEST_PATH_DATA[person1].keys():
            spotlight = SHORTEST_PATH_DATA[person1][person2]

            endpoints = [el[1][0] for el in spotlight]
            fastest_time_path = [el for el in spotlight if el[1][0] == min(endpoints)][
                0
            ]
            n1, n2, _, lhF, _, phF = fastest_time_path

            true_sp = [integer_lbl_to_static_node_map[v] for v in phF]
            sizesf, deltatf, redundanciesF = calc_size_of_single_path(
                hyperpath=true_sp, temporal_network=TS, dataset=dataset, option=option
            )

            fastest_time_path = [n1, n2, lhF, len(phF) - 1, phF, sizesf, redundanciesF]

            lengths = [el[2] for el in spotlight]
            shortest_len_path = [el for el in spotlight if el[2] == min(lengths)][0]
            n1, n2, lhS, _, phS, _ = shortest_len_path

            true_sp = [integer_lbl_to_static_node_map[v] for v in phS]
            sizess, deltats, redundanciesS = calc_size_of_single_path(
                hyperpath=true_sp, temporal_network=TS, dataset=dataset, option=option
            )

            tdiff = n2[0] - n1[0]

            shortest_len_path = [n1, n2, tdiff, lhS, phS, sizess, redundanciesS]

            data = {"F": fastest_time_path, "S": shortest_len_path, "equal": phS == phF}

            SHORTEST_PATH_DATA_enriched[person1][person2] = data

    return SHORTEST_PATH_DATA_enriched


def embed_time_series_for_dataset(temporal_network, dataset_name, root, verbose=False):
    warnings.warn(
        "embed_time_series_for_dataset(...) performs file I/O and is deprecated; "
        "use io_embed_time_series_for_dataset(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return io_embed_time_series_for_dataset(
        temporal_network, dataset_name, root, verbose=verbose
    )


def io_embed_time_series_for_dataset(
    temporal_network, dataset_name, root, verbose=False
):
    """
    Calculate the supra-adjacency representation of a temporal network

    Parameters
    ----------
    temporal_network : dict
        A dictionary of {time: hypergraph or graph} representing the temporal network
    dataset_name : str
        The name of the dataset, used to determine time difference calculation
    root : str
        The root directory to save the results
    verbose : bool
        Whether to print progress

    Returns
    -------
    static_node_to_integer_lbl_map : dict
        A dictionary mapping integer node labels to (time, node) tuples
    integer_lbl_to_static_node_map : dict
        A dictionary mapping (time, node) tuples to integer node labels
    V : list
        A list of unique individuals in the temporal network
    Vt0 : list
        A list of individuals active at the chosen time t0 in the temporal network

    """

    numwindows = get_ds_windowsize(nsamples=len(temporal_network))

    # Pick subset of times and calculate unique individuals
    if verbose:
        _log("2. PICK SUBSET OF TIMES")

    num_active_nodes = []
    unique_individuals = set()

    for t, H in temporal_network.items():
        num_active_nodes.append([t, H.num_nodes()])
        unique_individuals.update(H.get_nodes())

    num_active_nodes = pd.DataFrame(num_active_nodes, columns=["t", "|V|"])
    max_num_active_nodes_at_t = max(num_active_nodes["|V|"].unique())

    # pick the starting point as the first place where the number of active nodes is at maximum
    if numwindows == len(temporal_network):
        start_index = 0
    else:
        start_index = min(
            num_active_nodes[num_active_nodes["|V|"] == max_num_active_nodes_at_t].index
        )

    _log(
        f"Len of time-series = {len(temporal_network)}, numwindows={numwindows}, start_index={start_index}"
    )

    # Relabel appropriately using unique-individuals
    fname_TSmap = f"{root}/paths_temporal/relabeled_{dataset_name}_TS_and_mapping.pck"

    if verbose:
        _log("3. Relabelling the entire Time series".upper())
    remapping = {
        old: new
        for (old, new) in zip(unique_individuals, range(len(unique_individuals)))
    }
    for t, H in temporal_network.items():
        oldelist = H.get_edges()
        newelist = relabel_edges_with_mapping(edges=oldelist, mapping=remapping)
        temporal_network[t] = hgx.Hypergraph(newelist)

    pk.dump([temporal_network, remapping], open(fname_TSmap, "wb"))

    new_unique_individuals = set(remapping.values())

    if verbose:
        _log(f"Numwindows = {numwindows}")

    if verbose:
        _log("4. Pick subtimes".upper())
    # Pick subtimes
    subtimes = list(temporal_network.keys())[start_index : start_index + numwindows]

    # Create subTS of projected networks
    proj_subnetwork = {t: clique_projection(temporal_network[t]) for t in subtimes}

    # create subTS of true dyad networks
    dyad_subnetwork = {
        t: temporal_network[t].subhypergraph_by_orders(sizes=[2], keep_nodes=True)
        for t in subtimes
    }

    # create subTS of true only-ho-interactions networks

    higher_orders = {
        t: list(set(H.get_orders()) - {1}) for t, H in temporal_network.items()
    }
    ho_only_subnetwork = {
        t: temporal_network[t].subhypergraph_by_orders(
            orders=higher_orders[t], keep_nodes=True
        )
        for t in subtimes
    }
    proj_ho_only_subnetwork = {
        t: clique_projection(ho_only_subnetwork[t]) for t in subtimes
    }

    if verbose:
        _log("5. Embed using supra-adjacency".upper())
    # Embed HO
    G_HO_static_OG = supra_adj(
        temporal_network=proj_subnetwork,
        subtimes=subtimes,
        unique_individuals=unique_individuals,
        dataset=dataset_name,
    )
    # Embed DY
    G_DY_static_OG = supra_adj(
        temporal_network=dyad_subnetwork,
        subtimes=subtimes,
        unique_individuals=unique_individuals,
        dataset=dataset_name,
    )
    # Embed HO only
    G_only_HO_static_OG = supra_adj(
        temporal_network=proj_ho_only_subnetwork,
        subtimes=subtimes,
        unique_individuals=unique_individuals,
        dataset=dataset_name,
    )

    # Map (t,n)-nodes to new integer node label
    assert (set(G_DY_static_OG.nodes)).intersection(set(G_HO_static_OG.nodes)) == set(
        G_DY_static_OG.nodes
    ), "The dyad nodeset should be contained IN the HO nodeset (in the case where nx.empty_graph was used in supra-adj embedding)"

    assert (set(G_only_HO_static_OG.nodes)).intersection(
        set(G_HO_static_OG.nodes)
    ) == set(
        G_only_HO_static_OG.nodes
    ), "The ho-only nodeset should be contained IN the HO nodeset (in the case where nx.empty_graph was used in supra-adj embedding)"

    static_node_to_integer_lbl_map = {
        k: i
        for i, k in enumerate(
            set(G_HO_static_OG.nodes).union(set(G_DY_static_OG.nodes))
        )
    }
    # dict of {new_lbl: (t,n)}
    integer_lbl_to_static_node_map = {
        v: k for k, v in static_node_to_integer_lbl_map.items()
    }

    fname_map = f"{root}/paths_temporal/node_mappings.pck"
    pk.dump(
        [static_node_to_integer_lbl_map, integer_lbl_to_static_node_map],
        open(fname_map, "wb"),
    )

    t0 = subtimes[0]
    V = list(new_unique_individuals)
    Vt0 = temporal_network[t0].get_nodes()

    num_people = len(unique_individuals)

    fname = f"{root}/paths_temporal/V_Vt0_t0_numpeeps.pck"
    pk.dump([V, Vt0, t0, num_people], open(fname, "wb"))

    G_HO_static = nx.relabel_nodes(
        G_HO_static_OG, mapping=static_node_to_integer_lbl_map
    )
    G_DY_static = nx.relabel_nodes(
        G_DY_static_OG, mapping=static_node_to_integer_lbl_map
    )
    G_onlyHO_static = nx.relabel_nodes(
        G_only_HO_static_OG, mapping=static_node_to_integer_lbl_map
    )

    if verbose:
        _log("6. SAVE STATIC EMBEDDINGS \n")
    pk.dump(
        G_HO_static, open(f"{root}/paths_temporal/G_HO_Static_{dataset_name}.pck", "wb")
    )
    pk.dump(
        G_DY_static, open(f"{root}/paths_temporal/G_DY_Static_{dataset_name}.pck", "wb")
    )
    pk.dump(
        G_onlyHO_static,
        open(f"{root}/paths_temporal/G_HOonly_Static_{dataset_name}.pck", "wb"),
    )

    return static_node_to_integer_lbl_map, integer_lbl_to_static_node_map, V, Vt0


def HO_convert_node_labels_to_integers(H: hgx.Hypergraph) -> hgx.Hypergraph:
    oldelist = H.get_edges()

    mapping = {old: new for (old, new) in zip(H.get_nodes(), range(H.num_nodes()))}

    newelist = relabel_edges_with_mapping(edges=oldelist, mapping=mapping)

    Hnew = hgx.Hypergraph(newelist)
    return Hnew


def calc_redundancy_info(SQ_ARR):
    """
    Function to construct a dictionary with information on redundancy present in the paths

    Parameters
    ----------
    SQ_ARR : dict
        A dictionary of {person1: {person2: {'F': (n1, n2, tdiffF, lenhoF, pathhoF, sizesf, redundanciesF),
                        'S': (n1, n2, tdiffS, lenhoS, pathhoS, sizess, redundanciesS),
                        'equal': bool (pathhoS == pathhoF)
                       }
               }
    }
    where sizesf, sizess are lists of orders of each step in the path
    and redundanciesF, redundanciesS are lists of number of redundant edges at each step
    and tdiffF, tdiffS are the total time taken for the fastest/shortest path respectively
    'F' refers to fastest
    'S' refers to shortest

    Returns
    -------
    redundancy_df_S : pd.DataFrame
        A dataframe with redundancy information for shortest paths
    redundancy_df_F : pd.DataFrame
        A dataframe with redundancy information for fastest paths


    """

    perc_redund_for_single_path = lambda arr: (np.array(arr) != 0).sum() / len(arr)

    redundancy_df = pd.DataFrame(
        columns=[
            "perc_of_path_is_redund_S",
            "perc_of_path_is_redund_F",
            "avg_num_redund_S",
            "avg_num_redund_F",
            "lS",
            "lF",
        ]
    )
    for n1 in SQ_ARR.keys():
        for n2 in SQ_ARR[n1].keys():
            # print(f"{n1} -> {n2}", end='\r')
            _, _, _, lS, _, _, redS = SQ_ARR[n1][n2]["S"]
            _, _, _, lF, _, _, redF = SQ_ARR[n1][n2]["F"]

            # print(perc_redund_for_single_path(redS), perc_redund_for_single_path(redF))
            redundancy_df.loc[f"{n1}_{n2}"] = [
                perc_redund_for_single_path(redS),
                perc_redund_for_single_path(redF),
                np.array(redS).mean(),
                np.array(redF).mean(),
                # np.array(sizess).mean(), np.array(sizesf).mean(),
                lS,
                lF,
            ]

    dataframes = {}

    for opt in ["S", "F"]:
        df = (
            redundancy_df[
                [f"perc_of_path_is_redund_{opt}", f"avg_num_redund_{opt}", f"l{opt}"]
            ]
            .groupby(f"l{opt}")
            .agg(
                {
                    f"perc_of_path_is_redund_{opt}": ["mean", "std"],
                    f"avg_num_redund_{opt}": ["mean", "std"],
                }
            )
        )
        df.columns = [
            f"perc_of_path_is_redund_{opt}_mean",
            f"perc_of_path_is_redund_{opt}_std",
            f"avg_num_redund_{opt}_mean",
            f"avg_num_redund_{opt}_std",
        ]
        df = (
            df.reset_index()
            .join(
                redundancy_df.groupby(f"l{opt}").agg({f"l{opt}": "count"}),
                on=f"l{opt}",
                rsuffix="_sum",
            )
            .set_index(f"l{opt}")
            .sort_index()
        )
        dataframes[opt] = df

    # 2 columns perc and avg_num will be equal if only 0's/1's in path
    return dataframes["S"], dataframes["F"]


def save_redundancy_info(redundancy_df_S, redundancy_df_F, regime, dataset, root):
    warnings.warn(
        "save_redundancy_info(...) performs file I/O and is deprecated; "
        "use io_save_redundancy_info(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return io_save_redundancy_info(
        redundancy_df_S=redundancy_df_S,
        redundancy_df_F=redundancy_df_F,
        regime=regime,
        dataset=dataset,
        root=root,
    )


def io_save_redundancy_info(redundancy_df_S, redundancy_df_F, regime, dataset, root):
    """
    Function to save redundancy information dataframes

    """
    dsdir = f"{root}/paths_temporal"
    os.path.isdir(dsdir)
    os.makedirs(dsdir, exist_ok=True)

    fname = f"{dsdir}/redundancy_S_{dataset}_{regime}.pck"
    pk.dump(redundancy_df_S, open(fname, "wb"))
    _log(f"Saved {fname}")

    fname = f"{dsdir}/redundancy_F_{dataset}_{regime}.pck"
    pk.dump(redundancy_df_F, open(fname, "wb"))
    _log(f"Saved {fname}")


def P7_construct_square_arrays(SHORTEST_PATH_DATA, V, Vt0):
    """
    Function to construct square arrays from SHORTEST_PATH_DATA by extracting the relevant information
    SQ_ARR is of the form:
    {person1: {person2: {'F': (n1, n2, tdiffF, lenhoF, pathhoF, sizesf, redundanciesF),
                        'S': (n1, n2, tdiffS, lenhoS, pathhoS, sizess, redundanciesS),
                        'equal': bool (pathhoS == pathhoF)
                       }
               }

    """

    # 0th dimension <-- SPL and 1st dimension <-- FPL
    PATH_LEN_ARRAY = np.empty((len(V), len(V), 2))
    PATH_LEN_ARRAY.fill(np.nan)

    AVG_ORD_ARRAY = np.empty((len(V), len(V), 2))
    AVG_ORD_ARRAY.fill(np.nan)

    BEST_TIME_ARRAY = np.empty((len(V), len(V)))
    BEST_TIME_ARRAY.fill(np.nan)

    DYADIC_PROP_ARRAY = np.empty((len(V), len(V), 2))
    DYADIC_PROP_ARRAY.fill(np.nan)

    PATH_TIME_ARRAY = np.empty((len(V), len(V), 2))
    PATH_TIME_ARRAY.fill(np.nan)

    for person1 in tqdm(SHORTEST_PATH_DATA.keys()):
        for person2 in SHORTEST_PATH_DATA[person1].keys():
            spotlight = SHORTEST_PATH_DATA[person1][person2]

            #
            n1, n2, tdiffF, lhF, phF, sizesf, redundanciesF = spotlight["F"]
            n1, n2, tdiffS, lhS, phS, sizess, redundanciesS = spotlight["S"]
            #

            path_length = [lhS, lhF]
            avg_order = [np.mean(sizess), np.mean(sizesf)]
            best_time = spotlight["equal"]

            dyad_prop = [
                sum(el == 2 for el in sizess) / len(sizess),
                sum(el == 2 for el in sizesf) / len(sizesf),
            ]
            path_time = [tdiffS, tdiffF]

            PATH_LEN_ARRAY[person1][person2] = path_length
            AVG_ORD_ARRAY[person1][person2] = avg_order
            PATH_TIME_ARRAY[person1][person2] = path_time
            DYADIC_PROP_ARRAY[person1][person2] = dyad_prop
            BEST_TIME_ARRAY[person1][person2] = best_time

    outputs = (
        PATH_LEN_ARRAY[Vt0],
        AVG_ORD_ARRAY[Vt0],
        PATH_TIME_ARRAY[Vt0],
        DYADIC_PROP_ARRAY[Vt0],
        BEST_TIME_ARRAY[Vt0],
    )

    return outputs


def P8_save_square_arrays(outputs, option, regime, verbose, dsdir):
    warnings.warn(
        "P8_save_square_arrays(...) performs file I/O and is deprecated; "
        "use io_P8_save_square_arrays(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return io_P8_save_square_arrays(outputs, option, regime, verbose, dsdir)


def io_P8_save_square_arrays(outputs, option, regime, verbose, dsdir):
    (
        PATH_LEN_ARRAY_HO,
        AVG_ORD_ARRAY_HO,
        PATH_TIME_ARRAY_HO,
        DYADIC_PROP_ARRAY_HO,
        BEST_TIME_ARRAY_HO,
    ) = outputs

    # if verbose: print(f"len(Vt0) = {len(Vt0)} for {regime}")

    # SAVING ALL FILES
    arrnames = [
        "SPL",
        "FPL",  # Shortest-Path-Length, Fastest-Path-Length
        "STL",
        "FTL",  # Shortest-Time-Length, Fastest-Time-Length
        "SAO",
        "FAO",  # Shortest-Avg-Order, Fastest-Avg-Order
        "SPropDy",
        "FPropDy",  # Proportion-of-Dyadic-Steps-in-Shortest/Fastest-Path
        "BestTime",
    ]  # Whether-Shortest-and-Fastest-Path-are-the-Same-Timewise

    arrnames = [f"{el}_{regime}" for el in arrnames]

    arrays = [
        PATH_LEN_ARRAY_HO[:, :, 0],
        PATH_LEN_ARRAY_HO[:, :, 1],
        PATH_TIME_ARRAY_HO[:, :, 0],
        PATH_TIME_ARRAY_HO[:, :, 1],
        AVG_ORD_ARRAY_HO[:, :, 0],
        AVG_ORD_ARRAY_HO[:, :, 1],
        DYADIC_PROP_ARRAY_HO[:, :, 0],
        DYADIC_PROP_ARRAY_HO[:, :, 1],
        BEST_TIME_ARRAY_HO,
    ]

    for a, arrobj in zip(arrnames, arrays):
        fname = f"{dsdir}/paths_temporal/{a}_{option}.pck"
        if verbose:
            _log(f"Save {fname}")
        pk.dump(arrobj, open(fname, "wb"))


def calc_shortest_fastest_paths_temporal_hypergraphs(
    root=None, dataset_name=None, verbose=False, option="min", regime="HO"
):
    warnings.warn(
        "calc_shortest_fastest_paths_temporal_hypergraphs(...) performs extensive file I/O and is deprecated; "
        "use io_calc_shortest_fastest_paths_temporal_hypergraphs(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return io_calc_shortest_fastest_paths_temporal_hypergraphs(
        root=root,
        dataset_name=dataset_name,
        verbose=verbose,
        option=option,
        regime=regime,
    )


def io_calc_shortest_fastest_paths_temporal_hypergraphs(
    root=None, dataset_name=None, verbose=False, option="min", regime="HO"
):
    assert option in [
        "min",
        "max",
        "mean",
    ], "option must be one of ['min', 'max', 'mean']"
    assert regime in [
        "HO",
        "DY",
        "HOonly",
    ], "regime must be one of ['HO', 'DY', 'HOonly']"
    assert dataset_name is not None, "Please specify dataset_name"
    assert root is not None, "Please specify root directory"

    if root is None:
        _log("Please specify root directory")
        raise ValueError

    if verbose:
        _log("1. LOAD PRECALCULATED VARIABLES")

    fnamefile = f"{root}/paths_temporal/G_{regime}_Static_{dataset_name}.pck"
    gstatic = pk.load(open(fnamefile, "rb"))

    fname_map = f"{root}/paths_temporal/node_mappings.pck"
    static_node_to_integer_lbl_map, integer_lbl_to_static_node_map = pk.load(
        open(fname_map, "rb")
    )

    fname = f"{root}/paths_temporal/V_Vt0_t0_numpeeps.pck"
    V, Vt0, _, _ = pk.load(open(fname, "rb"))

    if verbose:
        _log("2. PREP HELPER VARIABLES")
    temp = pd.DataFrame(
        data=static_node_to_integer_lbl_map.keys(),
        index=static_node_to_integer_lbl_map.values(),
        columns=["times", "OG-node"],
    )
    temp["new-node"] = temp.index
    temp.sort_values("OG-node").head()

    mintime = temp.iloc[temp["times"].idxmin()]["times"]
    tydel = temp[temp["OG-node"].isin(Vt0) & temp["times"].isin([mintime])]
    start_node_labels = tydel["new-node"].tolist()
    tydel.sort_values("OG-node")

    if verbose:
        _log("3. LOAD TIME-SERIES")
    fname_TSmap = f"{root}/paths_temporal/relabeled_{dataset_name}_TS_and_mapping.pck"
    TS, _ = pk.load(open(fname_TSmap, "rb"))

    fname = f"{root}/paths_temporal/SQ_ARR_{regime}_{dataset_name}.pck"

    if os.path.exists(fname):
        if verbose:
            _log("4. LOAD PRECALCULATED SQUARE ARRAYS from P5")
        SHORTEST_PATH_DATA_after_P5 = pk.load(open(fname, "rb"))
    else:
        if verbose:
            _log("4. CALC SHORTEST/FASTEST PATHS")
        fdict, sdict = P4_calc_shortest_fastest_paths(
            verbose=verbose,
            start_node_labels=start_node_labels,
            static_graph_embedded=gstatic,
        )

        if verbose:
            _log("5. CALCULATE BEST PATHS")
        SHORTEST_PATH_DATA_after_P5 = P5_calc_best_paths(
            start_node_labels=start_node_labels,
            temp=temp,
            sdict=sdict,
            fdict=fdict,
            integer_lbl_to_static_node_map=integer_lbl_to_static_node_map,
        )

        pk.dump(SHORTEST_PATH_DATA_after_P5, open(fname, "wb"))

    if verbose:
        _log(
            f"6. FOR BEST PATHS: CALCULATE AVG ORDERS using {option.upper()} selection strategy"
        )
    SHORTEST_PATH_DATA_after_P6 = P6_calc_avg_orders(
        SHORTEST_PATH_DATA_after_P5,
        integer_lbl_to_static_node_map,
        TS,
        dataset_name,
        option=option,
    )

    if verbose:
        _log(f"6.5. CALCULATE REDUNDANCY INFO")
    redundancy_df_S, redundancy_df_F = calc_redundancy_info(SHORTEST_PATH_DATA_after_P6)
    io_save_redundancy_info(
        redundancy_df_S=redundancy_df_S,
        redundancy_df_F=redundancy_df_F,
        dataset=dataset_name,
        root=root,
        regime=regime,
    )

    if verbose:
        _log(
            f"7. FOR PATHS, CALCULATE SQUARE ARRAYS using {option.upper()} selection strategy"
        )
    processed_path_data_square_array = P7_construct_square_arrays(
        SHORTEST_PATH_DATA_after_P6, V, Vt0
    )

    if verbose:
        _log(f"8. SAVE SQUARE ARRAYS using {option.upper()} selection strategy")
    if verbose:
        _log(f"{option}")
    io_P8_save_square_arrays(
        processed_path_data_square_array,
        option="min",
        regime=regime,
        verbose=False,
        dsdir=root,
    )

    if verbose:
        _log("Done with script!\n")

    return processed_path_data_square_array
