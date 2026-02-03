import logging
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb, factorial

logger = logging.getLogger(__name__)


def _log(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    logger.info(message)


def lin_system(t, X, F, JF, JH, alpha, *params):
    dim = len(X) // 2
    X_s = X[:dim]
    Eta = X[dim:]

    JF_X_s = JF(X_s, *params)
    JH_X_s = JH(X_s)
    J_alpha = JF_X_s - alpha * JH_X_s

    new_X_s = F(0, X_s, *params)
    new_Eta = J_alpha.dot(Eta)

    return np.concatenate((new_X_s, new_Eta))


def lin_system_a2a(t, X, F, JF, sigmas, N, JHs, alpha, *params):
    dim = len(X) // 2
    X_s = X[:dim]
    Eta = X[dim:]

    JF_X_s = JF(X_s, *params)

    if not type(sigmas) == np.ndarray:
        sigmas = np.array(sigmas)
    rescaled_sigmas = sigmas / sigmas[0]
    all2all_weights = [
        factorial(N - 2) / factorial(N - 2 - d) for d in range(len(sigmas))
    ]
    weighted_JH_X_s = [
        sigma * w * JH(X_s)
        for sigma, w, JH in zip(rescaled_sigmas, all2all_weights, JHs)
    ]
    JH_X_s = sum(weighted_JH_X_s)

    J_alpha = JF_X_s - alpha * JH_X_s

    new_X_s = F(0, X_s, *params)
    new_Eta = J_alpha.dot(Eta)

    return np.concatenate((new_X_s, new_Eta))


def sprott_algorithm(
    alpha,
    C,
    F,
    JF,
    JH,
    Y0,
    params,
    integration_time=400.0,
    integration_step=0.01,
    verbose=True,
):
    """
    Evaluates the Master Stability Function as the maximum Lyapunov exponent using the Sprott's algorithm [1]

    Parameters
    ----------
    alpha: value for which the MSF is computed.
    C: number of cycles of the algorithm.
    F: function determining the dynamics of the isolated system.
        It is a callable as requested by scipy.solve_ivp
    JF: Jacobian matrix of the function f.
        It is a callable that returns the value of the Jacobian at a given point.
    JH: Jacobian matrix of the coupling function.
        It is a callable that returns the value of the Jacobian at a given point.
    Y0: initial condition of the isolated system, and initial perturbation.
        It is a list-like object.
    params: parameters of function f.
        It is a tuple of parameters used by the function F.
    integration_time: time over which the system is integrated in each cycle.
    integration_step: step of the integrating function.

    Returns
    -------
    interval
        interval of values over which the MSF is computed.
    MSF
        MSF evaluated over the interval of values selected.

    References
    ---------
    [1] J.C. Sprott, Chaos and Time-Series Analysis, Oxford University Press vol.69, pp.116-117 (2003).
    """
    dim = len(Y0) // 2
    Eta0 = Y0[dim:]
    Eta0_norm = np.linalg.norm(Eta0)

    lyap = np.zeros((C,))
    for iter in range(C):
        if verbose:
            _log("Integrating over cycle " + str(iter + 1) + " of " + str(C))
        sol = solve_ivp(
            fun=lin_system,
            t_span=[0.0, integration_time],
            t_eval=np.arange(0.0, integration_time, integration_step),
            y0=Y0,
            args=(F, JF, JH, alpha, *params),
            method="LSODA",
        )

        EtaT = sol.y[dim:, -1]
        EtaT_norm = np.linalg.norm(EtaT)

        lyap[iter] = np.log(EtaT_norm / Eta0_norm) / integration_time

        Eta0 = EtaT * Eta0_norm / EtaT_norm

        Y0[dim:] = Eta0

    return np.mean(lyap)


def sprott_algorithm_multi(
    alpha,
    C,
    F,
    JF,
    sigmas,
    N,
    JHs,
    Y0,
    params,
    all2all=True,
    integration_time=400.0,
    integration_step=0.01,
    verbose=True,
):
    """
    Evaluates the Master Stability Function as the maximum Lyapunov exponent using the Sprott's algorithm [1]

    Parameters
    ----------
    alpha: value for which the MSF is computed.
    C: number of cycles of the algorithm.
    F: function determining the dynamics of the isolated system.
        It is a callable as requested by scipy.solve_ivp
    JF: Jacobian matrix of the function f.
        It is a callable that returns the value of the Jacobian at a given point.
    JHs: Jacobian matrices of the coupling functions.
        It is a list of callables that return the value of the Jacobians at a given point.
    Y0: initial condition of the isolated system, and initial perturbation.
        It is a list-like object.
    params: parameters of function f.
        It is a tuple of parameters used by the function F.
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
    dim = len(Y0) // 2
    Eta0 = Y0[dim:]
    Eta0_norm = np.linalg.norm(Eta0)

    lyap = np.zeros((C,))
    for iter in range(C):
        if verbose:
            _log("Integrating over cycle " + str(iter + 1) + " of " + str(C))

        if all2all:
            sol = solve_ivp(
                fun=lin_system_a2a,
                t_span=[0.0, integration_time],
                t_eval=np.arange(0.0, integration_time, integration_step),
                y0=Y0,
                args=(F, JF, sigmas, N, JHs, alpha, *params),
                method="LSODA",
            )

        EtaT = sol.y[dim:, -1]
        EtaT_norm = np.linalg.norm(EtaT)

        lyap[iter] = np.log(EtaT_norm / Eta0_norm) / integration_time

        Eta0 = EtaT * Eta0_norm / EtaT_norm
        Eta0_norm = np.linalg.norm(Eta0)

        Y0[dim:] = Eta0

    return np.mean(lyap)


def is_natural_coupling(JHs, dim, verbose=True, *, seed: int | None = None, rng=None):
    orders = len(JHs)

    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)
    X = rng.random(size=(dim,))
    for d in range(orders - 1):
        JH1 = JHs[d]
        JH2 = JHs[d + 1]

        if (JH1(X) - JH2(X)).any():
            if verbose:
                _log("The coupling is not natural")
            return False

    if verbose:
        _log("The coupling is natural")
    return True


def is_all_to_all(hypergraph, verbose=True):
    if hypergraph.is_weighted():
        _log(
            "The higher-order network is weighted. Only unweighted higher-order networks are considered"
        )
        return False
    else:
        N = hypergraph.num_nodes()
        for order in range(1, hypergraph.max_order() + 1):
            num_edges = hypergraph.num_edges(order=order)
            if not comb(N, order + 1) == num_edges:
                if verbose:
                    _log("The higher-order network is not all-to-all")
                return False

        if verbose:
            _log("The higher-order network is all-to-all")
        return True
