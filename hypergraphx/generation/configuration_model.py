from collections import Counter
import logging

import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.exceptions import InvalidParameterError
from hypergraphx.generation._rng import np_rng, split_seed


def _build_hypergraph_from_sampled_edges(
    edges,
    *,
    duplicate_output="merge",
    weighted_output=False,
):
    edge_counts = Counter(tuple(sorted(edge)) for edge in edges)
    generated_edges = list(edge_counts.keys())

    if duplicate_output == "merge":
        new_h = Hypergraph(weighted=weighted_output)
        if weighted_output:
            new_h.add_edges(generated_edges, weights=[1] * len(generated_edges))
        else:
            new_h.add_edges(generated_edges)
        return new_h

    if duplicate_output == "count":
        new_h = Hypergraph(weighted=True)
        new_h.add_edges(
            generated_edges, weights=[edge_counts[edge] for edge in generated_edges]
        )
        return new_h

    if duplicate_output == "error":
        repeated_edges = [edge for edge, count in edge_counts.items() if count > 1]
        if repeated_edges:
            raise InvalidParameterError(
                "Repeated sampled hyperedges are not allowed when "
                "duplicate_output='error'."
            )
        new_h = Hypergraph(weighted=weighted_output)
        if weighted_output:
            new_h.add_edges(generated_edges, weights=[1] * len(generated_edges))
        else:
            new_h.add_edges(generated_edges)
        return new_h

    raise InvalidParameterError(
        "duplicate_output must be one of: 'merge', 'count', 'error'."
    )


def _cm_MCMC(
    hypergraph,
    n_steps=1000,
    label="edge",
    n_clash=1,
    restrict_to_same_size=True,
    duplicate_output="merge",
    *,
    seed: int | None = None,
):
    """
    Conduct Markov Chain Monte Carlo in order to approximately
    sample from the space of appropriately-labeled graphs.
    n_steps: number of steps to perform
    label: the label space to use. Can take values in ['vertex', 'edge'].
    n_clash: the number of clashes permitted when updating the edge counts in vertex-labeled MH.
        n_clash = 0 will be exact but very slow.
        n_clash >= 2 may lead to performance gains at the cost of decreased accuracy.
    restrict_to_same_size: if True, only rewire pairs of hyperedges with the same size
    duplicate_output: controls how repeated sampled hyperedges are handled
    """
    rng = np_rng(seed)

    def proposal_generator(m):
        # Propose a transition in edge-labeled MH.

        def __proposal(edge_list):
            i, j = rng.integers(0, m, 2)
            f1, f2 = edge_list[i], edge_list[j]
            if restrict_to_same_size:
                while len(f1) != len(f2):
                    i, j = rng.integers(0, m, 2)
                    f1, f2 = edge_list[i], edge_list[j]
            g1, g2 = __pairwise_reshuffle(f1, f2)
            return i, j, f1, f2, g1, g2

        return __proposal

    def __pairwise_reshuffle(f1, f2):
        # Randomly reshuffle the nodes of two edges while preserving their sizes.

        f = list(f1) + list(f2)
        intersection = set(f1).intersection(set(f2))
        ix = list(intersection)
        g1 = ix.copy()
        g2 = ix.copy()

        for v in ix:
            f.remove(v)
            f.remove(v)

        for v in f:
            if (len(g1) < len(f1)) & (len(g2) < len(f2)):
                if rng.random() < 0.5:
                    g1.append(v)
                else:
                    g2.append(v)
            elif len(g1) < len(f1):
                g1.append(v)
            elif len(g2) < len(f2):
                g2.append(v)
        if len(g1) != len(f1):
            logger = logging.getLogger(__name__)
            logger.warning("Inconsistent reshuffle sizes.")
            logger.debug("%s %s %s %s", f1, f2, g1, g2)
        return tuple(sorted(g1)), tuple(sorted(g2))

    def edge_mh(message=True):
        mh_rounds = 0
        mh_steps = 0
        c_new = [list(c) for c in hypergraph.get_edges()]
        m = len(c_new)

        proposal = proposal_generator(m)

        def mh_step():
            i, j, f1, f2, g1, g2 = proposal(c_new)
            c_new[i] = sorted(g1)
            c_new[j] = sorted(g2)

        n = 0

        while n < n_steps:
            mh_step()
            n += 1

        new_h = _build_hypergraph_from_sampled_edges(
            c_new,
            duplicate_output=duplicate_output,
            weighted_output=hypergraph.is_weighted(),
        )
        mh_steps += n
        mh_rounds += 1

        if message:
            logging.getLogger(__name__).info("%s steps completed.", n_steps)

        return new_h

    def vertex_labeled_mh(message=True):
        rand = rng.random
        randint = rng.integers

        k = 0
        done = False
        c = Counter(hypergraph._edge_list)

        epoch_num = 0
        n_rejected = 0

        m = sum(c.values())

        mh_rounds = 0
        mh_steps = 0

        while not done:
            # initialize epoch
            l = list(c.elements())

            add = []
            remove = []
            num_clash = 0
            epoch_num += 1

            # within each epoch

            k_rand = 20000  # generate many random numbers at a time

            k_ = 0
            ij = randint(0, m, k_rand)
            a = rand(k_rand)
            while True:
                if k_ >= k_rand / 2.0:
                    ij = randint(0, m, k_rand)
                    a = rand(k_rand)
                    k_ = 0
                i, j = (ij[k_], ij[k_ + 1])
                k_ += 2

                f1, f2 = l[i], l[j]
                while f1 == f2:
                    i, j = (ij[k_], ij[k_ + 1])
                    k_ += 2
                    f1, f2 = l[i], l[j]
                if restrict_to_same_size:
                    while len(f1) != len(f2):
                        i, j = (ij[k_], ij[k_ + 1])
                        k_ += 2
                        f1, f2 = l[i], l[j]
                        while f1 == f2:
                            i, j = (ij[k_], ij[k_ + 1])
                            k_ += 2
                            f1, f2 = l[i], l[j]

                inter = 2 ** (-len((set(f1).intersection(set(f2)))))
                if a[k_] > inter / (c[f1] * c[f2]):
                    n_rejected += 1
                    k += 1
                else:  # if proposal was accepted
                    g1, g2 = __pairwise_reshuffle(f1, f2)
                    num_clash += remove.count(f1) + remove.count(f2)
                    if (num_clash >= n_clash) & (n_clash >= 1):
                        break
                    else:
                        remove.append(f1)
                        remove.append(f2)
                        add.append(g1)
                        add.append(g2)
                        k += 1
                    if n_clash == 0:
                        break

            add = Counter(add)
            add.subtract(Counter(remove))

            c.update(add)
            done = k - n_rejected >= n_steps
        if message:
            logging.getLogger(__name__).info(
                "%s epochs completed, %s steps taken, %s steps rejected.",
                epoch_num,
                k - n_rejected,
                n_rejected,
            )

        new_h = _build_hypergraph_from_sampled_edges(
            list(c.elements()),
            duplicate_output=duplicate_output,
            weighted_output=hypergraph.is_weighted(),
        )
        mh_steps += k - n_rejected
        mh_rounds += 1
        return new_h

    if label == "edge":
        return edge_mh()
    elif label == "vertex":
        return vertex_labeled_mh()
    raise InvalidParameterError("label must be one of: 'edge', 'vertex'.")


def configuration_model(
    hypergraph,
    n_steps=1000,
    label="edge",
    order=None,
    size=None,
    n_clash=1,
    restrict_to_same_size=True,
    duplicate_output="merge",
    seed: int | None = None,
):
    """
    Sample a randomized hypergraph using a configuration-model-style MCMC.

    The sampler supports two labeling conventions:

    - ``label="edge"``: rewires pairs of hyperedges directly.
    - ``label="vertex"``: samples in the space of vertex-labeled hypergraphs
      using a collision-controlled update scheme.

    By default, all hyperedges are eligible for rewiring. If ``order`` or
    ``size`` is specified, only hyperedges of the selected size are rewired and
    all other hyperedges are copied unchanged into the output hypergraph.

    Parameters
    ----------
    hypergraph : Hypergraph
        Input hypergraph to randomize.
    n_steps : int, default=1000
        Number of MCMC update steps.
    label : {"edge", "vertex"}, default="edge"
        Labeling convention used by the sampler.
    order : int, optional
        Hyperedge order to rewire. If provided, only hyperedges of size
        ``order + 1`` are randomized.
    size : int, optional
        Hyperedge size to rewire. Mutually exclusive with ``order``.
    n_clash : int, default=1
        Collision threshold used in the vertex-labeled sampler. Only relevant
        when ``label="vertex"``.
    restrict_to_same_size : bool, default=True
        If ``True``, proposals are restricted to pairs of hyperedges with the
        same size, so rewiring is performed within size classes.
    duplicate_output : {"merge", "count", "error"}, default="merge"
        Controls how repeated sampled hyperedges are handled in the returned
        object. ``"merge"`` collapses duplicates into a simple hypergraph,
        ``"count"`` encodes multiplicities as edge weights, and ``"error"``
        raises if repeated sampled hyperedges occur.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Hypergraph
        Randomized hypergraph.

    Raises
    ------
    InvalidParameterError
        If both ``order`` and ``size`` are specified.
    InvalidParameterError
        If ``label`` is not one of ``"edge"`` or ``"vertex"``.
    InvalidParameterError
        If ``duplicate_output="count"`` is used with a weighted input
        hypergraph.

    Notes
    -----
    In the size-restricted mode, the function first extracts the corresponding
    subhypergraph, randomizes it, and then reinserts the untouched hyperedges.
    The default ``duplicate_output="merge"`` returns a simple hypergraph and
    collapses repeated sampled hyperedges.

    Examples
    --------
    >>> from hypergraphx import Hypergraph
    >>> from hypergraphx.generation import configuration_model
    >>> H = Hypergraph(edge_list=[(0, 1), (1, 2), (0, 1, 2)], weighted=False)
    >>> H2 = configuration_model(H, n_steps=10, label="edge", seed=0)
    >>> isinstance(H2, Hypergraph)
    True

    >>> H3 = configuration_model(H, n_steps=10, seed=0, duplicate_output="count")
    >>> H3.is_weighted()
    True
    """
    if order is not None and size is not None:
        raise InvalidParameterError("Only one of order and size can be specified.")
    if duplicate_output == "count" and hypergraph.is_weighted():
        raise InvalidParameterError(
            "duplicate_output='count' is only supported for unweighted hypergraphs."
        )
    if order is None and size is None:
        return _cm_MCMC(
            hypergraph,
            n_steps=n_steps,
            label=label,
            n_clash=n_clash,
            restrict_to_same_size=restrict_to_same_size,
            duplicate_output=duplicate_output,
            seed=seed,
        )

    if size is None:
        size = order + 1

    tmp_h = hypergraph.get_edges(
        size=size, up_to=False, subhypergraph=True, keep_isolated_nodes=True
    )
    # Derive a distinct seed for the size-restricted shuffle to keep deterministic behavior.
    seed_rng = np_rng(seed) if seed is not None else None
    sub_seed = split_seed(seed_rng) if seed_rng is not None else None
    shuffled = _cm_MCMC(
        tmp_h,
        n_steps=n_steps,
        label=label,
        n_clash=n_clash,
        restrict_to_same_size=restrict_to_same_size,
        duplicate_output=duplicate_output,
        seed=sub_seed,
    )
    for e in hypergraph.get_edges():
        if len(e) != size:
            if shuffled.is_weighted():
                weight = hypergraph.get_weight(e) if hypergraph.is_weighted() else 1
                shuffled.add_edge(e, weight=weight)
            else:
                shuffled.add_edge(e)
    return shuffled
