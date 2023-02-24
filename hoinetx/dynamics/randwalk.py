#%%
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from hoinetx.linalg.linalg import *


#%%

# define a function called transition_matrix that given an object HG of this type hoinetx.core.hypergraph.Hypergraph returns K
def transition_matrix(HG : Hypergraph)->sparse.spmatrix:
    '''Compute the transition matrix of the random walk on the hypergraph.
    
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
    '''
    IM = incidence_matrix(HG).todense()
    A  = IM @ IM.T
    #C = IM.T @ IM
    #C_hat is just the diagonal of C
    sizes = HG.get_sizes()
    # make a diagonal matrix with the sizes of the nodes
    C_hat = np.diag(sizes)
    K = IM @ C_hat @ IM.T - A
    # normalize K
    K = K / np.sum(K, axis=1)
    K = sparse.csr_matrix(K)
    return K