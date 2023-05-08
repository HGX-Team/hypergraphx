import numpy as np
from scipy import sparse

from hypergraphx import Hypergraph


def transition_matrix(HG: Hypergraph) -> sparse.spmatrix:
    """Compute the transition matrix of the random walk on the hypergraph.
    
    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.
        
    Returns
    -------
    K : np.ndarray
        
    The transition matrix of the random walk on the hypergraph.
    
    References
    ----------
    [1] Timoteo Carletti, Federico Battiston, Giulia Cencetti, and Duccio Fanelli, Random walks on hypergraphs, Phys. Rev. E 96, 012308 (2017)
    """
    N = HG.num_nodes()
    # assert if HG i connected
    assert HG.is_connected(), 'The hypergraph is not connected'
    hedge_list = HG.get_edges()
    T = np.zeros((N,N))
    for l in hedge_list:
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                T[l[i], l[j]] += len(l) - 1
                T[l[j], l[i]] += len(l) - 1
    # cast t to numpy.matrix
    # make it sparse

    T = np.matrix(T)   
    T = T / T.sum(axis=1)

    T = sparse.csr_matrix(T)
    return T


def random_walk(HG: Hypergraph, s: int, time: int) -> list:
    """Compute the random walk on the hypergraph.
    
    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.
    s : int
        The starting node of the random walk.
    time : int
        The number of steps of the random walk.
        
    Returns
    -------
    nodes : list
        The list of nodes visited by the random walk.
    """
    K = np.array(transition_matrix(HG).todense() )
    nodes = [s]
    for t in range(time):
        next_node = np.random.choice(K.shape[0], p=K[ nodes[-1],:])
        nodes.append(next_node)
    return nodes


def RW_stationary_state(HG: Hypergraph) -> np.ndarray:
    """Compute the stationary state of the random walk on the hypergraph.
    
    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.
        
    Returns
    -------
    stationary_state : np.ndarray
        The stationary state of the random walk on the hypergraph.
    """
    K = np.array(transition_matrix(HG).todense() )
    stationary_state = np.linalg.solve(np.eye(K.shape[0]) - K.T, np.ones(K.shape[0]))
    stationary_state = stationary_state / np.sum(stationary_state)
    return stationary_state


def random_walk_density(HG: Hypergraph, s: np.ndarray, time: int) -> list:
    """Compute the random walk on the hypergraph with starting density vector.
        
    Parameters
    ----------
    HG : Hypergraph
        The hypergraph on which the random walk is defined.
    s : np.ndarray
        The starting density vector of the random walk.
        
    Returns
    -------
    nodes : list
        The list of density vectors over time.
    """
    assert np.isclose(np.sum(s), 1), "The vector is not a probability density"
    
    K = np.array(transition_matrix(HG).todense() )
    starting_density = s
    density_list = [starting_density]
    for t in range(time):
        s = s @ K
        density_list.append(s)
    return density_list
