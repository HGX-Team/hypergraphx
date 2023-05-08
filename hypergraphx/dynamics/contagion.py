import numpy as np

from hypergraphx.utils.labeling import map_node, map_nodes


def simplicial_contagion(hypergraph, I_0, T, beta, beta_D, mu):
    """
    Simulates the contagion process on a simplicial hypergraph.
    The process is run for T time steps.
    The initial condition is given by I_0, which is a vector of length equal to the number of nodes in the hypergraph.
    The infection rate is beta, the three-body infection rate is beta_D, and the recovery rate is mu.
    The output is a vector of length T, where the i-th entry is the fraction of infected nodes at time i.

    Parameters
    ----------
    hypergraph : hypergraphx.Hypergraph
        The hypergraph on which the contagion process is run.

    I_0 : numpy.ndarray
        The initial condition of the contagion process.

    T : int
        The number of time steps.

    beta : float
        The infection rate.

    beta_D : float
        The three-body infection rate.

    mu : float
        The recovery rate.

    Returns
    -------
    numpy.ndarray
        The fraction of infected nodes at each time step.
    """
    
    numberInf = np.linspace(0, 0, T)
    Infected = np.sum(I_0)
    numberInf[0] = Infected
    N = len(I_0)
    nodes = hypergraph.get_nodes()
    mapping = hypergraph.get_mapping()
    I_old = np.copy(I_0)
    t = 1

    while np.sum(Infected)>0 and t<T:
        I_new = np.copy(I_old)

        # We run over the nodes
        for node in nodes:
            n = map_node(mapping, node)
            # if the node is susceptible, we run the infection process
            if I_old[n] == 0: 
                # we first run the two-body infections
                neighbors = hypergraph.get_neighbors(node, order=1)
                for neigh in neighbors:
                    m = map_node(mapping, neigh)
                    if I_old[m] == 1 and np.random.random() < beta:
                        I_new[n] = 1
                        break # if the susceptile node gets infected, we stop iterating over its neighbors
                if I_new[n] == 1: continue # if the susceptile node is already infected, we don't run the three-body processes
                # we run the three-body infections
                triplets = hypergraph.get_incident_edges(node, order=2)
                for triplet in triplets:
                    neighbors = list(triplet)
                    neighbors.remove(node)
                    m1,m2 = tuple(map_nodes(mapping,neighbors))
                    if I_old[m1] == 1 and I_old[m2] == 1 and np.random.random() < beta_D:
                        I_new[n] = 1
                        break # if the susceptile node gets infected, we stop iterating over the triplets
            # if the node is infected, we run the recovery process
            elif np.random.random() < mu:
                I_new[n] = 0
        
        I_old = np.copy(I_new)
        Infected =  np.sum(I_new)
        numberInf[t] = Infected
        t = t+1
    
    return numberInf / N
