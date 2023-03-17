import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from hnx.utils.labeling import *

def simplicial_contagion(hypergraph, I_0, T, beta, beta_D, mu):
    
    numberInf = np.linspace(0,0,T)
    numberInf[0] = np.sum(I_0)
    N = len(I_0)
    nodes = hypergraph.get_nodes()
    mapping = hypergraph.get_mapping()
    I_old = np.copy(I_0)
    t = 0

    while np.sum(Infected)>0 and t<T:
        t = t+1
        I_new = np.copy(I_old)

        # We run over the nodes
        for node in nodes:
            n = map_node(mapping, node)
            if I_old[n] == 0:
                neighbors = hypergraph.get_neighbors(node, order=1)
                for neigh in neighbors:
                    m = map_node(neigh)
                    if I_old[m] == 1 and np.random.random()[0] < beta:
                        I_new[n] = 1
                        break # check what it breaks
                # implement the simplicial contagion
            elif np.random.random()[0] < mu:
                I_new[n] == 0
        
        I_old = np.copy(I_new)
        numberInf[t] = np.sum(I_new)
    
    return numberInf/N
