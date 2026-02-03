from collections import Counter, defaultdict
import logging
from functools import reduce
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import binom


def _approximated_pvalue(t):
    """
    Return approximated p-value.

    Parameters
    ----------

    t   :tuple
        Tuple containing (N12,N,N1,N2,...Nn)

    Returns
    -------
    p-value:    float
                approximated p-value associated to the input tuple
    """
    n12 = t[0]
    n = t[1]
    ns = np.array(t[2:])
    order = len(ns)
    p = st.binom.sf(n12 - 1, p=np.prod(ns / n), n=n)
    if p >= 0:
        return p
    else:
        logging.getLogger(__name__).debug("%s %s %s", n12, n, ns)
        return p


def _get_bipartite_representation(hypergraph):
    def _as_int_multiplicity(w):
        if w is None:
            return 1
        if isinstance(w, bool):
            return int(w)
        if isinstance(w, (int, np.integer)):
            if w < 0:
                raise ValueError("Edge weights must be non-negative.")
            return int(w)
        if isinstance(w, (float, np.floating)):
            if w < 0:
                raise ValueError("Edge weights must be non-negative.")
            if float(w).is_integer():
                return int(w)
        raise ValueError(
            "Statistical filters require integer edge weights (multiplicity). "
            "If your hypergraph is weighted with real-valued weights, provide an unweighted hypergraph "
            "or integer-valued multiplicities."
        )

    edge_index = 0
    bipartite_list = []
    for edge in hypergraph.get_edges():
        w = hypergraph.get_weight(edge) if hypergraph.is_weighted() else 1
        for _ in range(_as_int_multiplicity(w)):
            for node in edge:
                bipartite_list.append((node, edge_index))
            edge_index += 1
    bipartite_df = pd.DataFrame(bipartite_list, columns=["a", "b"])
    return bipartite_df


def get_svh(
    hypergraph,
    max_order=10,
    alpha=0.01,
    mp: bool = False,
    n_jobs: int | None = None,
    max_tests_per_order: int | None = None,
):
    """
    Extract the Statistically Validated Hypergraph.

    Parameters
    -------------
    hypergraph:	Hypergraph
        Hypergraph object for which the SVH will be extracted.

    max_order:	int
        Maximum order of the hyperlinks to be tested

    alpha:		float
        Threshold of statistical significance for FDR validation.

    mp:			Bool (default: False)
        Specify whether to use multiprocessing or not.
    n_jobs:      int, optional
        Number of worker processes to use when `mp=True`. Defaults to `cpu_count()`.
    max_tests_per_order: int, optional
        Guardrail: maximum number of hypothesis tests per order. If exceeded, raises a ValueError.

    Returns
    -------------
    svh:		dict
        Dictionary where key is the order and value is a DataFrame with the result of validation. Each DataFrame is a Table with columns ['edge','pvalue','fdr'].
        'edge' contains all the hyperlinks (mapped as tuples) present in the hypergraph
        'pvalue' reports the pvalue
        'fdr' is a bool that is True if the hyperlink belongs to the SVH, False otherwise
    """

    df = _get_bipartite_representation(hypergraph)

    deg_set_b = df.groupby("b")["a"].count().reset_index()

    orders = deg_set_b.a.unique()
    orders = orders[(orders >= 2) & (orders <= max_order)]
    pvalues = {}

    for order in np.sort(orders):
        sub_deg = deg_set_b.query("a==@order").b.tolist()
        N = len(sub_deg)
        sub_edges = df.query("b in @sub_deg")
        tuples = (
            sub_edges.groupby("b")["a"]
            .apply(lambda x: tuple(sorted(x)))
            .unique()
            .tolist()
        )
        tuples_order = list(filter(lambda x: len(x) == order, tuples))
        neigh_set_a_sub = dict(
            sub_edges.groupby("a")["b"].apply(list).reset_index().values
        )

        tuples_params = [
            (
                len(
                    reduce(
                        lambda x, y: set(x).intersection(y),
                        [neigh_set_a_sub[node] for node in edge],
                    )
                ),
                N,
            )
            + tuple([len(neigh_set_a_sub[node]) for node in edge])
            for edge in tuples_order
        ]
        if max_tests_per_order is not None and len(tuples_params) > max_tests_per_order:
            raise ValueError(
                f"Too many hypothesis tests (order={order}): {len(tuples_params)} > {max_tests_per_order}. "
                "Reduce max_order, pre-filter your hypergraph, or increase max_tests_per_order."
            )

        if mp:
            p = Pool(processes=cpu_count() if n_jobs is None else n_jobs)
            pvalues[order] = dict(
                zip(tuples_order, p.map(_approximated_pvalue, tuples_params))
            )
            p.close()
        else:
            pvalues[order] = dict(
                zip(tuples_order, map(_approximated_pvalue, tuples_params))
            )

    svh = {}
    links = 0
    links_order = {}
    for order in sorted(pvalues):
        n_a = len(set(np.concatenate(list(pvalues[order].keys()))))
        n_possible = binom(n_a, order)
        bonf = alpha / n_possible

        temp_df = pd.DataFrame(pvalues[order].items())
        temp_df.columns = ["edge", "pvalue"]
        ps = np.sort(temp_df.pvalue)
        k = np.arange(1, len(ps) + 1) * bonf
        try:
            fdr = k[ps < k][-1]
        except:
            fdr = 0
        temp_df["fdr"] = temp_df["pvalue"] < fdr
        svh[order] = temp_df

    return svh


def get_svc(
    hypergraph,
    min_order=2,
    max_order=None,
    alpha=0.01,
    mp: bool = False,
    n_jobs: int | None = None,
    max_groups: int | None = None,
):
    """
    Extract the Statistically Validated Cores.

    Parameters
    -------------
    hypergraph:	Hypergraph
        Hypergraph object for which the SVS will be extracted.

    min_order:	int
        Minimum size of the hyperlinks to be tested

    max_order:	int
        Maximum size of the hyperlinks to be tested

    alpha:		float
        Threshold of statistical significance for FDR validation.
    mp:         bool (default: False)
        Specify whether to use multiprocessing or not.
    n_jobs:     int, optional
        Number of worker processes to use when `mp=True`. Defaults to `cpu_count()`.
    max_groups: int, optional
        Guardrail: maximum number of groups tested per order. If exceeded, raises a ValueError.

    Returns
    -------------
    svs:		DataFrame
        The DataFrame is a Table with columns ['edge','pvalue','fdr'].
        'group' contains all the cores (mapped as tuples) tested in the hypergraph
        'pvalue' reports the pvalue
        'fdr' is a bool that is True if the core has been validated, False otherwise
    """
    df = _get_bipartite_representation(hypergraph)

    observables = df.groupby("b")["a"].apply(lambda x: tuple(sorted(x))).tolist()

    if max_order:
        max_order = min(max_order, max(map(len, observables)))
    else:
        max_order = max(map(len, observables))

    s_groups = []
    neigh_set_a_sub = dict(df.groupby("a")["b"].apply(list).reset_index().values)
    N = df.b.nunique()
    na = df.a.nunique()

    svh_dfs = []

    for order in list(range(min_order, max_order + 1))[::-1]:
        drop = []
        for l in list(map(lambda x: tuple(combinations(x, order)), s_groups)):
            drop.extend(l)

        deg_a = Counter(df.a)
        groups = defaultdict(int)
        for l in map(
            lambda x: tuple(combinations(x, order)),
            filter(lambda x: len(x) >= order, observables),
        ):
            for g in l:
                if g not in drop:
                    groups[g] += 1

        tuples_params = [
            tuple([groups[i], N]) + tuple([deg_a[ii] for ii in i]) for i in groups
        ]
        if max_groups is not None and len(tuples_params) > max_groups:
            raise ValueError(
                f"Too many groups to test (order={order}): {len(tuples_params)} > {max_groups}. "
                "Increase max_groups, reduce max_order, or pre-filter your hypergraph."
            )
        if mp:
            p = Pool(processes=cpu_count() if n_jobs is None else n_jobs)
            pvalues = dict(zip(groups, p.map(_approximated_pvalue, tuples_params)))
            p.close()
        else:
            pvalues = dict(zip(groups, map(_approximated_pvalue, tuples_params)))

        n_possible = binom(na, order)
        bonf = alpha / n_possible

        temp_df = pd.DataFrame(pvalues.items())
        try:
            temp_df.columns = ["group", "pvalue"]
        except ValueError:
            temp_df = pd.DataFrame(columns=["group", "pvalue"])
        ps = np.sort(temp_df.pvalue)
        k = np.arange(1, len(ps) + 1) * bonf
        try:
            fdr = k[ps < k][-1]
        except IndexError:
            fdr = 0

        temp_df["w"] = temp_df.group.apply(lambda x: groups[x])
        temp_df["fdr"] = temp_df["pvalue"] < fdr

        svh_dfs.append(temp_df)  # .query('fdr'))

        s_groups_order = temp_df.query("fdr").group.tolist()

        s_groups.extend(s_groups_order)

    return pd.concat(svh_dfs)
