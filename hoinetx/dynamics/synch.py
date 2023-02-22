import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

from hoinetx.linalg.linalg import *
from utils import *

def MSF(F, JF, params, interval, JH, X0, integration_time = 2000.0, integration_step = 0.01, C = 5, verbose=True):
    """
    Evaluates the Master Stability Function

    Parameters
    ----------
    F: function determining the dynamics of the isolated system.
        It is a collable as requested by scipy.solve_ivp
    JF: Jacobian matrix of the function f.
        It is a callable that returns the value of the Jacobian at a given point.
    params: parameters of function f.
        It is a tuple of paramters used by the function F.
    interval: the interval of values over which the MSF is computed.
        It is a list-like object containing the values of alpha at which evaluating the MSF.
    JH: Jacobian matrix of the coupling function.
        It is a callable that returns the value of the Jacobian at a given point.
    X0: initial condition of the isolated system.
        It is a list-like object containg the initial conditions.
    integration_time: time over which the system is integrated.
    integration_step: step of the integrating function.
    C: number of cycles of the Sprott's algorithm.

    Returns
    -------
    interval: interval of values over which the MSF is computed.
    MSF: MSF evaluated over the interval of values selected.
    """

    # Here we make sure to be on the system attractor 
    if verbose: print("Getting to the attractor...")
    sol = solve_ivp(fun=F, t_span=[0.0,integration_time], t_eval=np.arange(0.0,integration_time,integration_step), y0=X0, 
                    args=params, method='LSODA')
    X0 = sol.y[:,-1]


    # Integrating the dynamics of the perturbation using Sprott's algorithm
    dim = len(X0)
    Eta0 = np.random.random(size=(dim,))*1e-9
    Eta0_norm = np.linalg.norm(Eta0)
    Y0 = np.concatenate((X0,Eta0))

    if verbose: print("Evaluating the Master Stability Function...")
    MSF = np.zeros(shape=len(interval))
    for i, alpha in enumerate(interval):
        MSF[i] = sprott_algorithm(alpha, C, F, JF, JH, Y0, Eta0_norm, params, integration_time/C, integration_step, verbose)

    return MSF
            
def higher_order_MSF(hypergraph, dim, F, JF, params, sigmas, JHs, X0, interval, diffusive_like=True, 
                    integration_time = 2000.0, integration_step = 0.01, C = 5, verbose=True):

    N = hypergraph.num_nodes()
    laplacians = laplacian_matrices_all_orders(hypergraph, weighted=True)

    # If the coupling is natural, we evaluate a single-parameter MSF for this scenario
    natural_coupling = is_natural_coupling(JHs,dim)
    if natural_coupling and diffusive_like:
        multiorder_laplacian = compute_multiorder_laplacian(laplacians, sigmas, degree_weighted=False)
        spectrum = sparse.linalg.eigsh(multiorder_laplacian, k=N, which='LM', return_eigenvectors=False)
    
        master_stability_function = MSF(F, JF, params, interval, JHs[0], X0, integration_time, integration_step, C, verbose)

        hypergraph_master_stability_function = MSF(F, JF, params, spectrum, JHs[0], X0, integration_time, integration_step, C, verbose)

        return master_stability_function, hypergraph_master_stability_function, spectrum

    # If the coupling is not natural but the Laplacian matrices commute, 
    # we check if the higher-order network is all-to-all
    #all2all = is_all_to_all(hypergraph)
    #if all2all:
    #    
    # If the coupling is not natural and the hypergraph is not all-to-all, no MSF can be calculated
    print("No Master Stability Function can be evaluated for this system.")

    return None