import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

def lin_system(t, X, F, JF, JH, alpha, *params):
    dim = len(X)//2
    X_s = X[:dim]
    Eta = X[dim:]

    JF_X_s = JF(X_s, *params)
    JH_X_s = JH(X_s)
    J_alpha = JF_X_s - alpha*JH_X_s

    new_X_s = F(0, X_s, *params)
    new_Eta = J_alpha.dot(Eta)

    return np.concatenate((new_X_s,new_Eta))

def sprott_algorithm(alpha, C, F, JF, JH, Y0, params, integration_time = 400.0, integration_step = 0.01, verbose = True):
    """
    Evaluates the Master Stability Function as the maximum Lyapunov exponenet using the Sprott's algorithm [1]

    Parameters
    ----------
    alpha: value for which the MSF is computed.
    C: number of cycles of the algorithm.
    F: function determining the dynamics of the isolated system.
        It is a collable as requested by scipy.solve_ivp
    JF: Jacobian matrix of the function f.
        It is a callable that returns the value of the Jacobian at a given point.
    JH: Jacobian matrix of the coupling function.
        It is a callable that returns the value of the Jacobian at a given point.
    Y0: initial condition of the isolated system, and initial perturbation.
        It is a list-like object.
    params: parameters of function f.
        It is a tuple of paramters used by the function F.
    integration_time: time over which the system is integrated in each cycle.
    integration_step: step of the integrating function.

    Returns
    -------
    interval: interval of values over which the MSF is computed.
    MSF: MSF evaluated over the interval of values selected.

    Reference
    ---------
    [1] J.C. Sprott, Chaos and Time-Series Analysis, Oxford University Press vol.69, pp.116-117 (2003).
    """
    dim = len(Y0)//2
    Eta0_norm = Y0[dim:]

    lyap = np.zeros((C,))
    for iter in C:
        if verbose: print("Integrating over cycle "+str(iter+1)+" of "+str(C))
        sol = solve_ivp(fun=lin_system, t_span=[0.0,integration_time], t_eval=np.arange(0.0,integration_time,integration_step), 
                        y0=Y0, args=(F, JF, JH, alpha, *params), method='LSODA')
        
        EtaT = sol.y[dim:,-1]
        EtaT_norm = np.linalg.norm(EtaT)

        lyap[iter] = np.log(EtaT_norm/Eta0_norm)/integration_time

        Eta0 = EtaT*Eta0_norm/EtaT_norm 

        Y0[dim:] = Eta0

    return np.mean(lyap)

def is_natural_coupling(JHs, dim, verbose = True):
    orders = len(JHs)

    X = np.random.random(size=(dim,))
    for d in range(orders-1):
        JH1 = JHs[d]
        JH2 = JHs[d+1]

        if not (JH1(X) - JH2(X)).any():
            if verbose: print("The coupling is not natural")
            return False 

    if verbose: print("The coupling is natural")
    return True
