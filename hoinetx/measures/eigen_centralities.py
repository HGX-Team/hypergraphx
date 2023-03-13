from numpy.linalg import norm
import numpy as np

def CEC_centrality(HG):
    '''
    Compute the CEC centrality for uniform hypergraphs.

    Parameters
    ----------
    
    HG : Hypergraph
        The uniform hypergraph on which the CEC centrality is computed.
    
    Returns
    -------
    cec : dict 
        The dictionary of keys nodes of HG and values the CEC centrality of the node.

        
    References
    ----------
    Three Hypergraph Eigenvector Centralities,
    Austin R. Benson,
    https://doi.org/10.1137/18M1203031  
    
    '''

    # check if the hypergraph is uniform, use raise exception
    if not HG.is_uniform():
        raise Exception("The hypergraph is not uniform.")
    # check if HG is connected, use raise exception
    if not HG.is_connected():
        raise Exception("The hypergraph is not connected.")
    # define W, matrix N x N where i,j is the number of common edges between i and j
    W = np.zeros((HG.num_nodes(), HG.num_nodes()))
    order = len(HG.get_edges()[0])
    for edge in HG.get_edges():
        for i in range(order):
            for j in range(i + 1, order):
                W[edge[i], edge[j]] += 1
                W[edge[j], edge[i]] += 1
    # compute the dominant eigenvector of W
    eigval, eigvec = np.linalg.eig(W)
    # return a dictionary of keys nodes of HG and values the corresponding element of the dominant eigenvector
    dominant_eig = eigvec[:, np.argmax(eigval)]
    return {node: dominant_eig[node] for node in range(HG.num_nodes())}

def ZEC_centrality(HG, max_iter=1000, tol=1e-7):
    '''
    Compute the ZEC centrality for uniform hypergraphs.

    Parameters
    ----------

    HG : Hypergraph
        The uniform hypergraph on which the ZEC centrality is computed.
    max_iter : int
        The maximum number of iterations.
    tol : float
        The tolerance for the stopping criterion.

    Returns
    -------
    ZEC : dict
        The dictionary of keys nodes of HG and values the ZEC centrality of the node.

    References
    ----------
    Three Hypergraph Eigenvector Centralities,
    Austin R. Benson,
    https://doi.org/10.1137/18M1203031

    '''
    if not HG.is_uniform():
        raise Exception("The hypergraph is not uniform.")

    if not HG.is_connected():
        raise Exception("The hypergraph is not connected.")

    g = lambda v, e: np.prod(v[list(e)])

    x = np.random.uniform(size=(HG.num_nodes()))
    x = x / norm(x, 1)

    for iter in range(max_iter):
        new_x = apply(HG, x, g)
        # multiply by the sign to try and enforce positivity
        new_x = np.sign(new_x[0]) * new_x / norm(new_x, 1)
        if norm(x - new_x) <= tol:
            break
        x = new_x.copy()
    else:
        "Iteration did not converge!"
    return {node: x[node] for node in range(HG.num_nodes())}

def HEC_centrality(HG, max_iter=100, tol=1e-6):
    '''
    
    Compute the HEC centrality for uniform hypergraphs.

    Parameters
    ----------

    HG : Hypergraph
        The uniform hypergraph on which the HEC centrality is computed.
    max_iter : int
        The maximum number of iterations.
    tol : float
        The tolerance for the stopping criterion.

    Returns
    -------
    HEC : dict
        The dictionary of keys nodes of HG and values the HEC centrality of the node.

    References
    ----------
    Three Hypergraph Eigenvector Centralities,
    Austin R. Benson,
    https://doi.org/10.1137/18M1203031
    
    '''
    # check if the hypergraph is uniform, use raise exception
    if not HG.is_uniform():
        raise Exception("The hypergraph is not uniform.")

    if not HG.is_connected():
        raise Exception("The hypergraph is not connected.")
    
    order = len(HG.get_edges()[0]) -1
    f = lambda v, m: np.power(v, 1.0 / m)
    g = lambda v, x: np.prod(v[list(x)])

    x = np.random.uniform(size=(HG.num_nodes()))
    x = x / norm(x, 1)

    for iter in range(max_iter):
        new_x = apply(HG, x, g)
        new_x = f(new_x, order)
        # multiply by the sign to try and enforce positivity
        #print(np.sign(new_x[0]) )
        new_x = np.sign(new_x[0]) * new_x / norm(new_x, 1)
        if norm(x - new_x) <= tol:
            break
        x = new_x.copy()
    else:
        print("Iteration did not converge!")
    return {node: x[node] for node in range(HG.num_nodes())}

def apply(HG, x, g=lambda v, e: np.sum(v[list(e)])):

    new_x = np.zeros(HG.num_nodes())
    for edge in HG.get_edges():
        edge = list(edge)
        #print(edge)
        # ordered permutations
        for shift in range(len(edge)):
            #print(shift)
            #print(edge[shift + 1 :] + edge[:shift])
            new_x[edge[shift]] += g(x, edge[shift + 1 :] + edge[:shift])
            #print(new_x)
    return new_x