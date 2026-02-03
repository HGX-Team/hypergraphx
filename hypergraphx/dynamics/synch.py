import logging
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

from hypergraphx.dynamics.utils import (
    is_all_to_all,
    is_natural_coupling,
    sprott_algorithm,
    sprott_algorithm_multi,
)
from hypergraphx.linalg.linalg import compute_multiorder_laplacian

logger = logging.getLogger(__name__)


def _log(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    logger.info(message)


def MSF(
    F,
    JF,
    params,
    interval,
    JH,
    X0,
    integration_time=2000.0,
    integration_step=0.01,
    C=5,
    verbose=True,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Evaluates the Master Stability Function

    Parameters
    ----------
    F: function determining the dynamics of the isolated system.
        It is a collable as requested by scipy.solve_ivp
    JF: Jacobian matrix of the function f.
        It is a callable that returns the value of the Jacobian at a given point.
    params: parameters of function f.
        It is a tuple of parameters used by the function F.
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
    MSF: MSF evaluated over the interval of values selected.
    """
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)

    # Here we make sure to be on the system attractor
    if verbose:
        _log("Getting to the attractor...")
    sol = solve_ivp(
        fun=F,
        t_span=[0.0, integration_time],
        t_eval=np.arange(0.0, integration_time, integration_step),
        y0=X0,
        args=params,
        method="LSODA",
    )
    X0 = sol.y[:, -1]

    # Integrating the dynamics of the perturbation using Sprott's algorithm
    dim = len(X0)
    Eta0 = rng.random(size=(dim,)) * 1e-9
    Eta0_norm = np.linalg.norm(Eta0)
    Y0 = np.concatenate((X0, Eta0))

    if verbose:
        _log("Evaluating the Master Stability Function...")
    MSF = np.zeros(shape=len(interval))
    for i, alpha in enumerate(interval):
        if verbose:
            _log("alpha = " + str(alpha))
        MSF[i] = sprott_algorithm(
            alpha,
            C,
            F,
            JF,
            JH,
            Y0,
            params,
            integration_time / C,
            integration_step,
            verbose,
        )

    return MSF


def MSF_multi_coupling(
    F,
    JF,
    params,
    interval,
    sigmas,
    N,
    JHs,
    X0,
    integration_time=2000.0,
    integration_step=0.01,
    C=5,
    verbose=True,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Evaluates the Master Stability Function for the higher-order all-to-all network

    Parameters
    ----------
    F: function determining the dynamics of the isolated system.
        It is a collable as requested by scipy.solve_ivp
    JF: Jacobian matrix of the function f.
        It is a callable that returns the value of the Jacobian at a given point.
    params: parameters of function f.
        It is a tuple of parameters used by the function F.
    interval: the interval of values over which the MSF is computed.
        It is a list-like object containing the values of alpha at which evaluating the MSF.
    sigmas: coupling strengths.
        It is a list-like object.
    N: number of nodes.
    JH: Jacobian matrix of the coupling function.
        It is a callable that returns the value of the Jacobian at a given point.
    X0: initial condition of the isolated system.
        It is a list-like object containg the initial conditions.
    integration_time: time over which the system is integrated.
    integration_step: step of the integrating function.
    C: number of cycles of the Sprott's algorithm.

    Returns
    -------
    MSF: MSF evaluated over the interval of values selected.
    """
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)

    # Here we make sure to be on the system attractor
    if verbose:
        _log("Getting to the attractor...")
    sol = solve_ivp(
        fun=F,
        t_span=[0.0, integration_time],
        t_eval=np.arange(0.0, integration_time, integration_step),
        y0=X0,
        args=params,
        method="LSODA",
    )
    X0 = sol.y[:, -1]

    # Integrating the dynamics of the perturbation using Sprott's algorithm
    dim = len(X0)
    Eta0 = rng.random(size=(dim,)) * 1e-9
    Eta0_norm = np.linalg.norm(Eta0)
    Y0 = np.concatenate((X0, Eta0))

    if verbose:
        _log("Evaluating the Master Stability Function...")
    MSF = np.zeros(shape=len(interval))
    for i, alpha in enumerate(interval):
        if verbose:
            _log("alpha = " + str(alpha))
        MSF[i] = sprott_algorithm_multi(
            alpha,
            C,
            F,
            JF,
            sigmas,
            N,
            JHs,
            Y0,
            params,
            True,
            integration_time / C,
            integration_step,
            verbose,
        )

    return MSF


def higher_order_MSF(
    hypergraph,
    dim,
    F,
    JF,
    params,
    sigmas,
    JHs,
    X0,
    interval,
    diffusive_like=True,
    integration_time=2000.0,
    integration_step=0.01,
    C=5,
    verbose=True,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    N = hypergraph.num_nodes()
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)

    # If the coupling is natural, we evaluate a single-parameter MSF for this scenario
    natural_coupling = is_natural_coupling(JHs, dim, verbose, rng=rng)
    if natural_coupling and diffusive_like:
        multiorder_laplacian = compute_multiorder_laplacian(
            hypergraph, sigmas, order_weighted=True, degree_weighted=False
        )
        spectrum = eigh(multiorder_laplacian.toarray(), eigvals_only=True)

        if verbose:
            _log("Starting the evaluation of the Master Stability Function...")
        master_stability_function = MSF(
            F,
            JF,
            params,
            interval,
            JHs[0],
            X0,
            integration_time,
            integration_step,
            C,
            verbose,
            rng=rng,
        )

        if verbose:
            _log(
                "Starting the evaluation of the Lyapunov exponents for the Laplacian eigenvalues..."
            )
        hon_master_stability_function = MSF(
            F,
            JF,
            params,
            spectrum[1:],
            JHs[0],
            X0,
            integration_time,
            integration_step,
            C,
            verbose,
            rng=rng,
        )

        return master_stability_function, hon_master_stability_function, spectrum

    # If the coupling is not natural but the Laplacian matrices commute,
    # we check if the higher-order network is all-to-all
    all2all = is_all_to_all(hypergraph, verbose)
    if all2all:
        master_stability_function = MSF_multi_coupling(
            F,
            JF,
            params,
            interval,
            sigmas,
            N,
            JHs,
            X0,
            integration_time,
            integration_step,
            C,
            verbose,
            rng=rng,
        )

        hon_master_stability_function = MSF_multi_coupling(
            F,
            JF,
            params,
            [sigmas[0] * N],
            sigmas,
            N,
            JHs,
            X0,
            integration_time,
            integration_step,
            C,
            verbose,
            rng=rng,
        )

        return master_stability_function, hon_master_stability_function, [sigmas[0] * N]

    # If the coupling is not natural and the hypergraph is not all-to-all, no MSF can be calculated
    _log("No Master Stability Function can be evaluated for this system.")

    return None
