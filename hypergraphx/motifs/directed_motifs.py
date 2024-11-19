
from hypergraphx import DirectedHypergraph
from hypergraphx.generation.directed_configuration_model import directed_configuration_model
from hypergraphx.motifs.utils import (
    _directed_motifs_ho_full,
    _directed_motifs_ho_not_full,
    directed_diff_sum, norm_vector,
)

def compute_directed_motifs(hypergraph: DirectedHypergraph, order=3, runs_config_model=10):
    """
    Compute the number of motifs of a given order in a directed hypergraph.

    Parameters
    ----------
    hypergraph : DirectedHypergraph
        The directed hypergraph of interest
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

    def _motifs_order_3(edges):
        full, visited = _directed_motifs_ho_full(edges, 3)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], full[i][1]))

        return res

    def _motifs_order_4(edges):
        full, visited = _directed_motifs_ho_full(edges, 4)
        not_full, visited = _directed_motifs_ho_not_full(edges, 4, visited)

        mappa={}
        for i in range(len(full)):
            mappa[full[i][0]]=full[i][1]
        for i in range(len(not_full)):
            mappa[not_full[i][0]]=not_full[i][1]
                
        res = []
        for key in mappa.keys():
            res.append((key,mappa[key]))
        return res
    
    
    edges = hypergraph.get_edges(size=order, up_to=True)
    output = {}

    print("Computing observed motifs of order {}...".format(order))

    if order == 3:
        output['observed'] = _motifs_order_3(edges)
        
    elif order == 4:
        output['observed'] = _motifs_order_4(edges)
    else:
        raise ValueError("Exact computation of motifs of order > 5 is not available.")
    
    if runs_config_model == 0:
        return output
            
    ROUNDS = runs_config_model

    results = []

    for i in range(ROUNDS):
        print("Computing config model motifs of order {}. Step: {}".format(order, i+1))
        e1 = directed_configuration_model(hypergraph).get_edges()
        
        if order == 3:
            m1 = _motifs_order_3(e1)
        elif order == 4:
            m1 = _motifs_order_4(e1)
        else:
            raise ValueError("Exact computation of motifs of order > 5 is not available.")
        
        results.append(m1)

    output['config_model'] = results
    
    delta = list(directed_diff_sum(output['observed'], output['config_model']))
    norm_delta = list(norm_vector(delta))
    output['norm_delta'] = []

    for i in range(len(delta)):
        output['norm_delta'].append((output['observed'][i][0], norm_delta[i]))

    return output