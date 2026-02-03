import logging

from hypergraphx import Hypergraph
from hypergraphx.generation.configuration_model import configuration_model
from hypergraphx.motifs.utils import (
    _motifs_ho_full,
    _motifs_ho_not_full,
    _motifs_standard,
    diff_sum,
    norm_vector,
)


def compute_motifs(
    hypergraph: Hypergraph,
    order=3,
    runs_config_model=10,
    *,
    seed: int | None = None,
    rng=None,
):
    """
    Compute the number of motifs of a given order in a hypergraph.

    Parameters
    ----------
    hypergraph : Hypergraph
        The hypergraph of interest
    order : int
        The order of the motifs to compute
    runs_config_model : int
        The number of runs of the configuration model

    Returns
    -------
    dict
        keys: 'observed', 'config_model', 'norm_delta'
        'observed' reports the number of occurrences of each motif in the observed hypergraph
        'config_model' reports the number of occurrences of each motif in each sample of the configuration model
        'norm_delta' reports the norm of the difference between the observed and the configuration model

    """
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    import numpy as np

    rng = rng if rng is not None else np.random.default_rng(seed)

    def _motifs_order_3(edges):
        full, visited = _motifs_ho_full(edges, 3)
        standard = _motifs_standard(edges, 3, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max(full[i][1], standard[i][1])))

        return res

    def _motifs_order_4(edges):
        full, visited = _motifs_ho_full(edges, 4)
        not_full, visited = _motifs_ho_not_full(edges, 4, visited)
        standard = _motifs_standard(edges, 4, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max([full[i][1], not_full[i][1], standard[i][1]])))

        return res

    edges = hypergraph.get_edges(size=order, up_to=True)
    output = {}

    logger = logging.getLogger(__name__)
    logger.info("Computing observed motifs of order %s...", order)

    if order == 3:
        output["observed"] = _motifs_order_3(edges)
    elif order == 4:
        output["observed"] = _motifs_order_4(edges)
    else:
        raise ValueError("Exact computation of motifs of order > 4 is not available.")

    if runs_config_model == 0:
        return output

    STEPS = hypergraph.num_edges(size=order, up_to=True) * 10
    ROUNDS = runs_config_model

    results = []

    for i in range(ROUNDS):
        logger.info("Computing config model motifs of order %s. Step: %s", order, i + 1)
        sub_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        e1 = configuration_model(hypergraph, label="stub", n_steps=STEPS, seed=sub_seed)
        if order == 3:
            m1 = _motifs_order_3(e1.get_edges())
        elif order == 4:
            m1 = _motifs_order_4(e1.get_edges())
        else:
            raise ValueError(
                "Exact computation of motifs of order > 4 is not available."
            )
        results.append(m1)

    output["config_model"] = results

    delta = list(diff_sum(output["observed"], output["config_model"]))
    norm_delta = list(norm_vector(delta))
    output["norm_delta"] = []

    for i in range(len(delta)):
        output["norm_delta"].append((output["observed"][i][0], norm_delta[i]))

    return output
