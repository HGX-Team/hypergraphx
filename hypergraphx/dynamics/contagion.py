import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from hypergraphx.utils.labeling import *

def simplicial_contagion(hypergraph, I_0, T, beta, beta_D, mu):
    
    numberInf = np.linspace(0,0,T)
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
    
    return numberInf/N
