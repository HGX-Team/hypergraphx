from hypergraphx.core.hypergraph import Hypergraph
from hypergraphx.motifs.utils import _motifs_ho_full, _motifs_ho_not_full, _motifs_standard


def compute_motifs(hypergraph: Hypergraph, order=3):
    """
    Compute the number of motifs of a given order in a hypergraph.

    Parameters
    ----------
    hypergraph : Hypergraph
        The hypergraph of interest
    order : int
        The order of the motifs to compute

    Returns
    -------
    list
        The list of motifs of the given order with their number of occurrences
    """
    edges = hypergraph.get_edges()

    def _motifs_order_3():
        full, visited = _motifs_ho_full(edges, 3)
        standard = _motifs_standard(edges, 3, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max(full[i][1], standard[i][1])))

        return res

    def _motifs_order_4():
        full, visited = _motifs_ho_full(edges, 4)
        not_full, visited = _motifs_ho_not_full(edges, 4, visited)
        standard = _motifs_standard(edges, 4, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max([full[i][1], not_full[i][1], standard[i][1]])))

        return res

    if order == 3:
        return _motifs_order_3()
    elif order == 4:
        return _motifs_order_4()
    else:
        print("Exact computation of motifs of order > 4 is not available.")

    """
    STEPS = len(edges) * 10
    ROUNDS = 10

    results = []

    for i in range(ROUNDS):
        e1 = hypergraph(edges)
        e1.MH(label='stub', n_steps=STEPS)
        if N == 3:
            m1 = motifs_order_3(e1.C, i)
        elif N == 4:
            m1 = motifs_order_4(e1.C, i)
        results.append(m1)

    output['config_model'] = results

    delta = diff_sum(output['motifs'], output['config_model'])
    norm_delta = norm_vector(delta)

    print(norm_delta)
    """
