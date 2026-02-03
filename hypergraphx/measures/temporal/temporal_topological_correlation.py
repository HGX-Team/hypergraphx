import itertools
import logging
from collections import Counter
from time import sleep

import networkx as nx
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover

    def tqdm(it, **kwargs):
        return it


from hypergraphx.representations import projections

logger = logging.getLogger(__name__)


def _to_df(H):
    """
    Convert a Temporal HyperGraph H to a DataFrame.

    Args:
        H: The input Temporal HyperGraph.

    Returns:
        df (pd.DataFrame): A DataFrame containing timestamp, nodes, and order columns.
    """
    # Create a DataFrame from the edges of the HyperGraph
    df = pd.DataFrame(H.get_edges(), columns=["timestamp", "nodes"])
    # Add a column 'order' to the DataFrame based on the length of 'nodes'
    df["order"] = df["nodes"].apply(lambda x: len(x))
    return df


def compute_all_nodes_shortest_path(G):
    """
    Compute the shortest path lengths between all pairs of nodes in a graph.

    Args:
        G: The input graph.

    Returns:
        dict: A dictionary containing node pairs as keys and their corresponding shortest path lengths as values.
    """
    if nx.is_connected(G):
        # If the graph is connected, use networkx's shortest_path_length function
        return dict(nx.shortest_path_length(G))
    else:
        raise TypeError("G must be connected")


def compute_all_edges_shortest_path(
    H, graph_distance=None, verbose=False, aggregate=False
):
    """
    Compute the shortest path lengths between all pairs of edges in a (temporal) HyperGraph.

    Args:
        H: The input (temporal) HyperGraph.
        graph_distance: Precomputed node shortest path distances (optional).
        verbose: If True, print progress information.
        aggregate: If True, assume H as a time-aggregated hypergraph; otherwise, aggregate the temporal hypergraph.

    Returns:
        dict: A dictionary containing edge pairs as keys and their corresponding shortest path lengths as values.
    """
    if aggregate:
        H_agg = H
    else:
        H_agg = H.aggregate(max(H.edges) + 100)[0]

    if graph_distance is None:
        if verbose:
            logger.info(
                "Computing node shortest path distance of the aggregated projected network..."
            )
        # Compute node shortest path distances for the aggregated projected network
        graph_distance = compute_all_nodes_shortest_path(
            projections.clique_projection(H_agg)
        )
        if verbose:
            logger.info("Complete!")

    if verbose:
        logger.info(
            "Computing topological distance of hyperlinks path distance of the higher order aggregated network..."
        )

    E = len([x for x in H_agg.get_edges()])
    edge_dic_distance = {}
    for edge1 in H_agg.get_edges():
        for edge2 in H_agg.get_edges():
            if (edge1, edge2) not in edge_dic_distance:
                if edge1 == edge2:
                    edge_dic_distance[(edge1, edge2)] = 0
                else:
                    # Compute the shortest path length between two edges based on node shortest path distances
                    edge_dic_distance[(edge1, edge2)] = (
                        min(
                            [
                                graph_distance[x][y]
                                for x, y in itertools.product(edge1, edge2)
                            ]
                        )
                        + 1
                    )
                    edge_dic_distance[(edge2, edge1)] = edge_dic_distance[
                        (edge1, edge2)
                    ]
    assert len(edge_dic_distance) == E**2
    if verbose:
        logger.info("Complete!")

    return edge_dic_distance


def get_mean_distance_events(H, order, edge_distance=None, cross_order=False):
    """
    Compute the mean distance between events in a HyperGraph.

    Args:
        H: The input HyperGraph.
        order: The order d of the events.
        edge_distance: Precomputed edge shortest path distances (optional).
        cross_order: If True, compute distances between events of different orders (one of order d and one of a different order d')

    Returns:
        cnt_ev1: A Counter object containing distances as keys and counts as values.
    """
    if edge_distance is None:
        edge_distance = compute_all_edges_shortest_path(H)

    cnt_ev1 = Counter()

    if cross_order:
        size = Counter([edge for edge in H.get_edges()])
        # events of order d
        size_order = {edge: size[edge] for edge in size.keys() if len(edge[1]) == order}
        # events of order d'
        size_other_order = {
            edge: size[edge] for edge in size.keys() if len(edge[1]) != order
        }
        for x, y in itertools.product(
            list(size_order.keys()), list(size_other_order.keys())
        ):
            # Compute distances between hyperlinks of different order once and update counts with the product of their weights
            cnt_ev1.update({edge_distance[(x, y)]: size[x] * size[y]})
        # Check that the total number of values in the counter is the total number of couples of events of different order

        assert sum(list(cnt_ev1.values())) == sum(list(size_order.values())) * sum(
            list(size_other_order.values())
        )

    else:
        size = Counter([edge for edge in H.get_edges() if len(edge[1]) == order])
        for x, y in itertools.combinations(list(size.keys()), 2):
            # Compute distances between hyperlinks of same order once and update counts with the product of their weights
            cnt_ev1.update({edge_distance[(x, y)]: size[x] * size[y]})
        # Compute the number of possible combinations of events of the same hyperlink
        cnt_ev1[0] = sum(list(map(lambda x: (x * (x - 1)) / 2, list(size.values()))))
        n = len([x for x in H.get_edges() if len(x[1]) == order])
        # Check that the total number of values in the counter is the total number of couples of events of same order
        assert sum(list(cnt_ev1.values())) == (n * (n - 1)) / 2
    return cnt_ev1


def topological_temporal_distance_diff_order(
    H, df, order, distance_dict, dt_list, skip_inf=True, verbose=False
):
    """
    Compute topological distances among events of different orders, given the temporal delay and the reference order.

    Args:
        H: The input Temporal HyperGraph.
        df: DataFrame representation of the temporal hypergraph.
        order: The reference order.
        distance_dict: Dictionary containing hyperlink topological distance information.
        dt_list: List of time delays.
        skip_inf: If False, include the overall topological distribution of events.
        verbose: If True, print progress information.

    Returns:
        tot_dic_dst_counter_gen: A dictionary containing the overall topological distances among events of different orders,
        and the one conditioned to their temporal delays.
    """
    # Introduce t_inf to check convergence to 1 of E[eta| delta t]
    t_inf = (np.max(df.timestamp) - np.min(df.timestamp)) + 100
    dt_list = dt_list + [t_inf]

    # Initialize dictionaries to store distances and counters
    tot_dic_dst_counter = {}
    tot_dic_dst_counter_gen = {}

    # Compute average topological distance of events of reference order with events of different orders
    cnt_ev1 = get_mean_distance_events(H, order, distance_dict, cross_order=True)
    tot_dic_dst_counter_gen["overall_avg_dist"] = np.average(
        list(cnt_ev1.keys()), weights=list(cnt_ev1.values())
    )
    n5 = df.shape[0]

    # Divide the events into those with reference order and those with different orders
    n2 = df[df.order == order].shape[0]
    n_other = df[df.order != order].shape[0]
    assert n2 * n_other == sum(list(cnt_ev1.values()))

    cnt_dst = Counter()
    cnt_spk = Counter()
    n0 = 0
    n_tot = 0

    # Iterate over time intervals
    for p, delta_t in enumerate(dt_list):
        # Use tqdm to check the progress
        if verbose:
            tqdm.write(str(delta_t))
            sleep(0.5)
        max_time = max(df.timestamp)
        index_set = set()
        t_cnt = 0
        e_cnt = 0
        i = 0

        if verbose:
            ref_list = tqdm(list(df.timestamp.unique()), position=0, leave=True)
        else:
            ref_list = list(df.timestamp.unique())

        # Iterate over reference times
        for ref_time in ref_list:
            # Introduce delta_t_low as the previous element in dt_list.
            # If delta_t is the first element, then delta_t_low=0
            if p == 0:
                delta_t_low = 0
            else:
                delta_t_low = dt_list[p - 1]
            t_cnt += 1

            # To speed up computation, update the cnt_dist dynamically.
            # At each consecutive delta_t in dt_list, update the counter including the couples of events
            # with temporal delay delta_t', such that delta_t_low <= delta_t' < delta_t. Each couple is computed once.

            # The original set of events includes events at time t and those with delay delta_t',
            # such that delta_t_low <= delta_t' < delta_t.
            df_rest = df[
                (df.timestamp == ref_time)
                | (
                    (df.timestamp >= ref_time + delta_t_low)
                    & (df.timestamp < ref_time + delta_t)
                )
            ]

            # Divide events into reference order and other orders
            df_rest_order = df_rest[df_rest.order == order]
            df_rest_other_order = df_rest[df_rest.order != order]

            # Divide events according to order and time in which they occur

            # 1)Events occurring at the reference time
            ref_nodes_order = df_rest_order[df_rest_order.timestamp == ref_time].nodes
            ref_nodes_other_order = df_rest_other_order[
                df_rest_other_order.timestamp == ref_time
            ].nodes

            # 2)Events occurring with delay delta_t' where delta_t_low <= delta_t' < delta_t
            df_rest_order = df_rest_order[df_rest_order.timestamp > ref_time]
            df_rest_other_order = df_rest_other_order[
                df_rest_other_order.timestamp > ref_time
            ]

            # Speed up the process by counting the number of events associated with the same hyperlink
            size_n_order = df_rest_order.groupby(df_rest_order.nodes).size()
            size_n_other_order = df_rest_other_order.groupby(
                df_rest_other_order.nodes
            ).size()
            # Concatenate sizes since no events in size_n_order are included in size_n_other_order
            size = pd.concat([size_n_order, size_n_other_order]).to_dict()

            # Check the total number of couples of events where one has the reference order
            # and the other has a different order for consistency checks
            n_tot += ref_nodes_order.shape[0] * df_rest_other_order.shape[0]
            n_tot += ref_nodes_other_order.shape[0] * df_rest_order.shape[0]
            # If delta_t_low is the smallest in the list, include also events occurring at the same timestamp
            if p == 0:
                n_tot += ref_nodes_order.shape[0] * (ref_nodes_other_order.shape[0])
                for x, y in itertools.product(ref_nodes_order, ref_nodes_other_order):
                    cnt_dst.update({distance_dict[(x, y)]: 1})

            # Update distance dict adding to each distance value the number of corresponding couples of events
            for x, y in itertools.product(
                ref_nodes_order, df_rest_other_order.nodes.unique()
            ):
                cnt_dst.update({distance_dict[(x, y)]: size[y]})
            for x, y in itertools.product(
                ref_nodes_other_order, df_rest_order.nodes.unique()
            ):
                cnt_dst.update({distance_dict[(x, y)]: size[y]})

            e_cnt += 1
            i += 1

        # Check the number of couples of events
        assert sum(list(cnt_dst.values())) == n_tot

        # Check convergence: lim delta_t -> inf E[eta|delta_t] = E[eta]
        if delta_t == t_inf:
            assert (
                np.average(list(cnt_dst.keys()), weights=list(cnt_dst.values()))
                == tot_dic_dst_counter_gen["overall_avg_dist"]
            )
            if skip_inf:
                continue

        tot_dic_dst_counter[delta_t] = cnt_dst.copy()

        assert e_cnt == df.timestamp.unique().shape[0]
        assert t_cnt == e_cnt

    tot_dic_dst_counter_gen["cond_dist"] = tot_dic_dst_counter.copy()

    return tot_dic_dst_counter_gen


def topological_temporal_distance_same_order(
    H, df, order, distance_dict, dt_list, skip_inf=True, verbose=False
):
    """
    Compute topological distances among events of the same order, given the temporal delay and the reference order.

    Args:
        H: The input Temporal HyperGraph.
        df: DataFrame representation of the temporal hypergraph.
        order: The reference order.
        distance_dict: Dictionary containing hyperlink topological distance information.
        dt_list: List of time delays.
        skip_inf: If False, include the overall topological distribution of events.
        verbose: If True, print progress information.

    Returns:
        tot_dic_dst_counter_gen: A dictionary containing the overall topological distances among events of the same order,
        and the one conditioned to their temporal delays.
    """
    tot_dic_dst_counter = {}
    tot_dic_dst_counter_gen = {}

    # Filter the DataFrame to retain events of the specified order
    df = df[df.order == order]

    # Introduce t_inf to check convergence to 1 of E[tau]
    t_inf = (np.max(df.timestamp) - np.min(df.timestamp)) + 100
    dt_list = dt_list + [t_inf]

    # Compute average topological distance of events of the same order with each other
    cnt_ev1 = get_mean_distance_events(H, order, distance_dict, cross_order=False)
    tot_dic_dst_counter_gen["overall_avg_dist"] = np.average(
        list(cnt_ev1.keys()), weights=list(cnt_ev1.values())
    )
    n5 = df.shape[0]

    # Check that the total number of pairs of contacts is equal to the total number of possible pairs of a set of df.shape[0] elements
    assert (n5 * (n5 - 1)) / 2 == sum(list(cnt_ev1.values()))

    cnt_dst = Counter()
    cnt_spk = Counter()
    n0 = 0
    n_tot = 0

    for p, delta_t in enumerate(dt_list):
        if verbose:
            tqdm.write(str(delta_t))
            sleep(0.5)
        max_time = max(df.timestamp)
        index_set = set()
        t_cnt = 0
        e_cnt = 0
        i = 0

        if verbose:
            ref_list = tqdm(list(df.timestamp.unique()), position=0, leave=True)
        else:
            ref_list = list(df.timestamp.unique())

        for ref_time in ref_list:
            # Introduce delta_t_low as the previous element in dt_list.
            # If delta_t is the first element, then delta_t_low=0
            if p == 0:
                delta_t_low = 0
            else:
                delta_t_low = dt_list[p - 1]
            t_cnt += 1

            if ref_time <= max_time:
                # To speed up computation, update the cnt_dist dynamically.
                # At each consecutive delta_t in dt_list, update the counter including the couples of events
                # with temporal delay delta_t', such that delta_t_low <= delta_t' < delta_t. Each couple is computed once.

                # The original set of events includes events at time t and those with delay delta_t',
                # such that delta_t_low <= delta_t' < delta_t.
                df_rest = df[
                    (df.timestamp == ref_time)
                    | (
                        (df.timestamp >= ref_time + delta_t_low)
                        & (df.timestamp < ref_time + delta_t)
                    )
                ]

                # Divide events according to the time in which they occur

                # 1)Events occurring at the reference time
                ref_nodes = df_rest[df_rest.timestamp == ref_time].nodes

                # 2)Events occurring with delay delta_t' where delta_t_low <= delta_t' < delta_t
                df_rest = df_rest[df_rest.timestamp > ref_time]

                # Speed up the process by counting the number of events associated with the same hyperlink

                size_n = df_rest.groupby(df.nodes).size()
                size = size_n.to_dict()
                # Compute the total number of couples of events with same reference order
                n_tot += ref_nodes.shape[0] * (sum(size_n.values))

                # If delta_t_low is the smallest in the list, include also events occurring at the same timestamp
                if p == 0:
                    n_tot += ((ref_nodes.shape[0] * (ref_nodes.shape[0] - 1))) / 2.0
                    for x, y in itertools.combinations(ref_nodes, 2):
                        cnt_dst.update({distance_dict[(x, y)]: 1})

                # Update distance dict adding to each distance value the number of corresponding couples of events
                for x, y in itertools.product(ref_nodes, df_rest.nodes.unique()):
                    cnt_dst.update({distance_dict[(x, y)]: size[y]})

                e_cnt += 1
                i += 1

        # Check the number of couples of events
        assert sum(list(cnt_dst.values())) == n_tot

        # Check convergence: lim delta_t -> inf E[eta|delta_t] = E[eta]
        if delta_t == t_inf:
            assert (
                np.average(list(cnt_dst.keys()), weights=list(cnt_dst.values()))
                == tot_dic_dst_counter_gen["overall_avg_dist"]
            )
            assert cnt_dst == cnt_ev1
            if skip_inf:
                continue

        tot_dic_dst_counter[delta_t] = cnt_dst.copy()
        assert e_cnt == df.timestamp.unique().shape[0]
        assert t_cnt == e_cnt

    tot_dic_dst_counter_gen["cond_dist"] = tot_dic_dst_counter.copy()

    return tot_dic_dst_counter_gen


def topological_temporal_cond_distance(
    H,
    order,
    distance_dict=None,
    same_order=True,
    fit_correlation=True,
    drop_duplicates=True,
    dt_list=None,
    skip_inf=True,
    verbose=False,
):
    """
    Compute topological distances among events of the same or different orders, given the temporal delay and the reference order.

    Args:
        H: The input Temporal HyperGraph.
        order: The reference order.
        distance_dict: Dictionary containing hyperlink topological distance information.
        same_order: If True, compute distances among events of the same order; otherwise, compute distances among events of different orders.
        fit_correlation: If True, compute the fit of the increasing trend of the avg normalized conditional topological distance.
        drop_duplicates: If True, remove duplicate entries in the DataFrame.
        dt_list: List of time delays.
        skip_inf: If False, include the overall topological distribution of events.
        verbose: If True, print progress information.


    Returns:
        A dictionary with the following keys:
            avg_top_dist: float
            the average topological distance among all considered events

            cond_top_dist_distribution: dict = {k:v}

            k is an element of dt_list
            v is the counter of the relative topological distances between events with temporal delay smaller than k

            avg_cond_top_dist: dict = {k,v}
            k is an element of dt_list
            v is the average topological conditional distance of events with delay smaller than k

            temp_top_corr: float
            fit of the increasing trend of the normalized average conditional topological distance as a function of the logarithm of the delay
    """

    # If distance_dict is not provided, compute it
    gen_dict = {}
    if distance_dict is None:
        distance_dict = compute_all_edges_shortest_path(H, verbose=verbose)

    # Create a DataFrame from the input Temporal HyperGraph and sort it by timestamp
    df = (
        _to_df(H)
        .sort_values("timestamp", ascending=True)
        .reset_index()
        .drop("index", axis=1)
    )

    # Remove duplicate entries in the DataFrame if drop_duplicates is True
    if drop_duplicates:
        df.drop_duplicates(["nodes", "timestamp"], inplace=True)

    # Ensure dt_list is a list
    if not isinstance(dt_list, list):
        raise TypeError("dt_list must be a list!")

    # Normalize timestamps by subtracting the minimum timestamp value
    df["timestamp"] = df.timestamp.values - min(df.timestamp.values)

    # Based on the same_order flag, call the appropriate function to compute distances
    if same_order:
        cond_distr = topological_temporal_distance_same_order(
            H, df, order, distance_dict, dt_list, skip_inf, verbose=verbose
        )
    else:
        cond_distr = topological_temporal_distance_diff_order(
            H, df, order, distance_dict, dt_list, skip_inf, verbose=verbose
        )
    gen_dict["avg_top_dist"] = cond_distr["overall_avg_dist"]
    gen_dict["cond_top_dist_distribution"] = cond_distr["cond_dist"]
    gen_dict["avg_cond_top_dist"] = {
        k: np.average(list(v.keys()), weights=list(v.values()))
        / gen_dict["avg_top_dist"]
        for k, v in cond_distr["cond_dist"].items()
    }
    # Compute the topological temporal correlation as a fit of the increasing trend of the normalized conditioned topopoligical distance

    if fit_correlation:
        avg_cond_top_dist_sr = pd.Series(gen_dict["avg_cond_top_dist"])
        min_val_idx = min(enumerate(avg_cond_top_dist_sr), key=lambda x: x[1])[0]
        avg_cond_top_dist_sr = avg_cond_top_dist_sr.iloc[min_val_idx:]
        if len(avg_cond_top_dist_sr) < 3:
            logger.info("fit with too few data points")
        coef = np.polyfit(
            np.log(avg_cond_top_dist_sr.index), avg_cond_top_dist_sr.values, 1
        )
        gen_dict["temp_top_corr"] = coef[0]
    return gen_dict
