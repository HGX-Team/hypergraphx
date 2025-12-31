import os
import time
from typing import List, Optional, Tuple, Union

import logging
import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import comb

from hypergraphx import Hypergraph
from hypergraphx.communities.hy_sc.model import HySC
from hypergraphx.linalg.linalg import binary_incidence_matrix, incidence_matrix

DEFAULT_SEED = 10
DEFAULT_INF = 1e10  # infinite initial value for the log-likelihood
DEFAULT_EPS = 1e-20  # epsilon for numerical stability


class HypergraphMT:
    """Implementation of the Hypergraph-MT probabilistic model from

    "Inference of hyperedges and overlapping communities in hypergraphs",
    Contisciani M., Battiston F., De Bacco C.

    The probabilistic generative model infers overlapping communities in hypergraphs.
    It is a mixed-membership model where we assume an assortative structure. The inference is performed
    using an efficient expectation-maximization (EM) algorithm that exploits the sparsity of the network,
    leading to an efficient and scalable implementation.
    """

    def __init__(
        self,
        noise_input_par: float = 0.001,
        min_value_par: float = 1e-5,
        max_value_par: float = 1e2,
        n_realizations: int = 10,
        max_iter: int = 500,
        check_convergence_every: int = 1,
        tolerance: float = 0.1,
        threshold_for_convergence: int = 15,
        verbose: bool = True,
    ) -> None:
        """Initialize the probabilistic model.

        Parameters
        ----------
        noise_input_par: noise to initialize the membership matrix u or the affinity matrix w around input values.
        min_value_par: minimum value for the parameters.
        max_value_par: minimum value for the parameters.
        n_realizations: number of realizations with different random initialization.
        max_iter: maximum number of EM iteration steps before aborting.
        check_convergence_every: number of steps in between every convergence check.
        tolerance: tolerance parameter for convergence of the log-likelihood.
        threshold_for_convergence: number of consecutive convergences for the EM to stop.
        verbose: flag to print details.
        """
        # Parameters related attributes.
        self.noise_input_par = noise_input_par
        self.min_value_par = min_value_par
        self.max_value_par = max_value_par
        # Training related attributes.
        self.n_realizations = n_realizations
        self.max_iter = max_iter
        self.check_convergence_every = check_convergence_every
        self.tolerance = tolerance
        self.threshold_for_convergence = threshold_for_convergence
        # Attribute to print details.
        self.verbose = verbose

        # Initial value for the maximum log-likelihood.
        self.maxL = -DEFAULT_INF
        # DataFrame to keep track of training information.
        self.train_info = pd.DataFrame()

    def fit(
        self,
        hypergraph: Hypergraph,
        K: int,
        seed: Optional[int] = None,
        normalizeU: bool = False,
        baseline_r0: bool = True,
        **extra_params,
    ) -> Tuple[np.array, np.array, float]:
        """Perform community detection on hypergraphs with a mixed-membership probabilistic model.

        Parameters
        ----------
        hypergraph: the hypergraph to perform inference on.
        K: number of communities.
        seed: random seed.
        normalizeU: if True, then the membership matrix u is normalized such that every row sums to 1.
        baseline_r0: if True, then for the first iteration u is initialized around the solution of the Hypergraph Spectral Clustering.
        **extra_params: additional keyword arguments handed to __check_fit_params to handle u and w.

        Returns
        -------
        u_f: membership matrix of dimension (N, K).
        w_f: affinity matrix of dimension (D-1, K).
        maxL: maximum log-likelihood value.
        """
        # Initialize all the values needed for training.
        self._check_fit_params(
            hypergraph=hypergraph,
            K=K,
            seed=seed,
            normalizeU=normalizeU,
            baseline_r0=baseline_r0,
            **extra_params,
        )

        # Keep track of log-likelihood values, running time, and other training info.
        train_info = []
        final_it, final_convergence = None, None

        for r in range(self.n_realizations):
            # Initialize psiOmega and psiBarOmega.
            self._initialize_psiOmega()

            # Initialize the membership matrix u and the affinity matrix w.
            # For the first iteration, we initialize u around the solution of the Hypergraph Spectral Clustering.
            # For the next ones, we initialize the parameters either randomly or
            # around the input values chosen with "initialize_u0".
            # In the end, we choose the realization with the best likelihood.
            if r == 0:
                self._initialize_u_w(
                    hyperEdges=self.hyperEdges, baseline_HySC=self.baseline_r0
                )
            else:
                self._initialize_u_w(hyperEdges=self.hyperEdges, baseline_HySC=False)

            # First update of the matrices u, psiOmega and psiBarOmega.
            self._initial_update_u_psi(r=r)
            # Initialize the rho matrix that represents the variational distribution used in the EM routine.
            self._update_rho()

            if self.verbose:
                _log(f"Updating realization {r} ...")
            # Initial value for the log-likelihood.
            loglik = -DEFAULT_INF
            # Convergence local variables.
            n_tolerance_reached = (
                0  # number of consecutive times the tolerance is reached
            )
            converged = False  # flag for reached convergence
            it = 0  # iteration

            # EM routine.
            while not converged and it < self.max_iter:
                time_start = time.time()
                # Train.
                self._update_em()

                # Check for convergence.
                loglik, n_tolerance_reached, converged = self._check_for_convergence(
                    it, loglik, n_tolerance_reached, converged
                )
                # Store training information.
                runtime = time.time() - time_start
                if not it % self.check_convergence_every:
                    train_info.append((r, self.seed, it, loglik, runtime, converged))
                it += 1

            if self.verbose:
                _log(f"N_real={r} -- num it={it} -- Loglikelihood:{loglik}")

            # Save parameters for the realization with the highest log-likelihood.
            if self.maxL < loglik:
                self._update_optimal_parameters()
                self.maxL = loglik
                final_it = it
                final_convergence = converged

            # Update seed.
            self._set_seed(self.seed + self.prng.randint(1, 1e6))
        # end cycle over realizations

        # Update DataFrame with training information.
        cols = [
            "realization",
            "seed",
            "iter",
            "loglik",
            "runtime",
            "reached_convergence",
        ]
        self.train_info = pd.DataFrame(train_info, columns=cols)

        # Convergence not reached.
        if np.logical_and(final_it == self.max_iter, not final_convergence):
            _log(f"Solution failed to converge in {self.max_iter} EM steps!")

        # Save inferred parameters.
        if self.out_inference:
            self._output_results(final_it)

        return self.u_f, self.w_f, self.maxL

    def _check_fit_params(
        self,
        hypergraph: Hypergraph,
        K: int,
        seed: Optional[int] = None,
        normalizeU: bool = False,
        baseline_r0: bool = True,
        **extra_params,
    ) -> None:
        """Pre-process the data and initialize parameters for the inference.

        Parameters
        ----------
        hypergraph: the hypergraph to perform inference on.
        K: number of communities.
        seed: random seed.
        normalizeU: if True, then the membership matrix u is normalized such that every row sums to 1.
        """
        # Save hypergraph.
        self.hypergraph = hypergraph

        # Set the pseudo-random number generator.
        self._set_seed(seed)

        # Weights of hyperedges.
        self.hye_weights = np.array(hypergraph.get_weights())
        # Hyperedges list.
        self.hyperEdges = np.array(hypergraph.get_edges(), dtype=object)
        # Weighted incidence matrix.
        self.incidence = incidence_matrix(hypergraph)
        # Binary incidence matrix.
        self.binary_incidence = binary_incidence_matrix(hypergraph)

        # Number of nodes, and number of hyperedges.
        self.N, self.E = self.incidence.shape
        # Number of communities.
        self.K = K
        # Maximum observed hyperedge size.
        self.D = max([len(e) for e in self.hyperEdges])

        node_order = list(hypergraph.get_mapping().classes_)
        # List of length N containing the indices of non-zero hyperedges for every node.
        self.hye_per_node = hypergraph.incident_edges_by_node(
            index_by="position", node_order=node_order
        )
        # List of list containing the indices of hyperedges with a given degree.
        edges_by_size = hypergraph.edges_by_size(index_by="position")
        self.HyD2eId = [edges_by_size.get(d, []) for d in range(2, self.D + 1)]
        # Hyperedges' size.
        self.HyeId2D = np.array(
            hypergraph.get_sizes()
        )  # TODO: check whether we want to refactor the name of this variable

        self.isolates = hypergraph.isolates(node_order=node_order)
        self.non_isolates = hypergraph.non_isolates(node_order=node_order)

        # Normalize u such that every row sums to 1.
        self.normalizeU = normalizeU

        # Initialize u around the solution of the Hypergraph Spectral Clustering for the first iteration.
        self.baseline_r0 = baseline_r0

        available_extra_params = [
            "fix_communities",  # flag to keep the communities fixed
            "fix_w",  # flag to keep the affinity matrix fixed
            "gammaU",  # constant to regularize the communities
            "gammaW",  # constant to regularize the affinity matrix
            # initialize u with input array, stored file, or the solution of the Hypergraph Spectral Clustering
            "initialize_u0",
            "initialize_w0",  # initialize w with input array or from a stored file
            "out_inference",  # flag to store the inferred parameters
            "out_folder",  # path to store the output
            "end_file",  # output file suffix
        ]
        for extra_param in extra_params:
            if extra_param not in available_extra_params:
                msg = "Ignoring unrecognised parameter %s." % extra_param
                raise Warning(msg)

        if "fix_communities" in extra_params:
            self.fix_communities = extra_params["fix_communities"]
        else:
            self.fix_communities = False

        if "fix_w" in extra_params:
            self.fix_w = extra_params["fix_w"]
        else:
            self.fix_w = False

        if "gammaU" in extra_params:
            self.gammaU = extra_params["gammaU"]
        else:
            self.gammaU = 0

        if "gammaW" in extra_params:
            self.gammaW = extra_params["gammaW"]
        else:
            self.gammaW = 0

        u0 = None
        if "initialize_u0" in extra_params:
            if extra_params["initialize_u0"] is not None:
                u0 = self._initialize_u0(extra_params["initialize_u0"])
        self.u0 = u0

        w0 = None
        if "initialize_w0" in extra_params:
            if extra_params["initialize_w0"] is not None:
                w0 = self._initialize_w0(extra_params["initialize_w0"])
        self.w0 = w0

        if "out_inference" in extra_params:
            self.out_inference = extra_params["out_inference"]
        else:
            self.out_inference = False

        if "out_folder" in extra_params:
            self.out_folder = extra_params["out_folder"]
        else:
            self.out_folder = "../data/output/"

        if "end_file" in extra_params:
            self.end_file = extra_params["end_file"]
        else:
            self.end_file = ""

    def _set_seed(self, seed: Optional[int]) -> None:
        """Set the container for the Mersenne Twister pseudo-random number generator."""
        if seed is None:
            seed = DEFAULT_SEED
        self.seed = seed
        self.prng = np.random.RandomState(seed)

    def _initialize_u0(self, input_val: Union[np.array, str]) -> np.array:
        """Initialize u with input array, stored file, or the solution of the Hypergraph Spectral Clustering."""
        # Initialize u with input array.
        if isinstance(input_val, np.ndarray):
            if input_val.shape == (self.N, self.K):
                u0 = u0_w0_from_nparray(input_val)
            else:
                msg = f"u0 must have shape {(self.N, self.K)}. In input was given {input_val.shape}!"
                raise ValueError(msg)
        # Initialize u either with the solution of the Hypergraph Spectral Clustering or from a stored file.
        elif isinstance(input_val, str):
            if input_val == "spectral":
                u0 = calculate_u_HySC(self.hypergraph, self.K, seed=self.seed)
            else:
                if os.path.exists(input_val):
                    u0 = u0_w0_from_file(input_val, par="u")
                else:
                    msg = f"{input_val} does not exist!"
                    raise ValueError(msg)
        return u0

    def _initialize_w0(self, input_val: Union[np.array, str]) -> np.array:
        """Initialize w with input array or from a stored file."""
        # Initialize w with input array.
        if isinstance(input_val, np.ndarray):
            if input_val.shape == (self.D - 1, self.K):
                w0 = u0_w0_from_nparray(input_val)
            else:
                msg = f"w0 must have shape {(self.D - 1, self.K)}. In input was given {input_val.shape}!"
                raise ValueError(msg)
        # Initialize w from a stored file.
        elif isinstance(input_val, str):
            if os.path.exists(input_val):
                w0 = u0_w0_from_file(input_val, par="w")
            else:
                msg = f"{input_val} does not exist!"
                raise ValueError(msg)
        return w0

    def _initialize_psiOmega(self) -> None:
        """Initialize psi matrices psiOmega and psiBarOmega, i.e., psi(0)(Omega^d, k) and psi(0)(BarOmega^d, k).
        They have dimension DxK, and the first row refers to degree=1. See the Supplementary Note 1 for details.
        """
        self.psiBarOmega = np.zeros((self.D, self.K))
        self.psiOmega = np.zeros((self.D, self.K))

        self._set_dummy_u0()
        u0 = np.copy(self.u0_dummy)
        u0_mean = np.mean(self.u0_dummy[self.non_isolates], axis=0)

        for k in range(self.K):
            Nk = np.count_nonzero(u0[:, k])
            self.psiOmega[0, k] = np.sum(u0[:, k])
            for d in range(1, self.D):
                self.psiOmega[d, k] = np.power(u0_mean[k], d + 1) * comb(Nk, d + 1)

    def _set_dummy_u0(self) -> None:
        """Initial dummy u0 to compute psiOmega in closed-form at step t==0."""
        uk = self.prng.random_sample(self.K)
        self.u0_dummy = np.tile(uk, [self.N, 1])

        row_sums = self.u0_dummy.sum(axis=1)
        self.u0_dummy[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _initialize_u_w(
        self, hyperEdges: np.array, baseline_HySC: bool = False
    ) -> None:
        """Initialization of the parameters u and w.

        Parameters
        ----------
        hyperEdges : array of length E, containing the sets of hyper-edges (as tuples).
        baseline_HySC : flag to initialize u around the solution of the Hypergraph Spectral Clustering.
        """
        # Initialize u around the solution of the Hypergraph Spectral Clustering.
        if baseline_HySC:
            if self.verbose:
                _log(
                    "u is initialized around the solution of the Hypergraph Spectral Clustering."
                )
            self.u0_current_real_t0 = calculate_u_HySC(
                self.hypergraph, self.K, seed=self.seed
            )
            max_entry = np.max(self.u0_current_real_t0)
            self.u0_current_real_t0 += (
                max_entry
                * self.noise_input_par
                * self.prng.random_sample(self.u0_current_real_t0.shape)
            )
        # Initialize u either randomly or around the input values chosen with "initialize_u0".
        else:
            if self.u0 is None:
                if self.verbose:
                    _log("u is initialized randomly.")
                self.u0_current_real_t0 = self._randomize_u0()
            else:
                if self.verbose:
                    _log(
                        f"u is initialized around the input values chosen with 'initialize_u0'."
                    )
                self.u0_current_real_t0 = self._add_noise_input(self.u0)
        # Initialize w either randomly or around the input values chosen with "initialize_w0".
        if self.w0 is None:
            if self.verbose:
                _log("w is initialized randomly.")
            self.w = self._randomize_w0(hyperEdges=hyperEdges)
        else:
            if self.verbose:
                _log(
                    f"w is initialized around the input values chosen with 'initialize_w0'."
                )
            self.w = self._add_noise_input(self.w0)
        self.w_old = np.copy(self.w)

    def _randomize_u0(self) -> np.array:
        """Initialize membership matrix u randomly."""
        u0 = self.prng.random_sample((self.N, self.K))
        row_sums = u0.sum(axis=1)
        u0[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        return u0

    def _randomize_w0(self, hyperEdges: Optional[np.array] = None) -> np.array:
        """Initialize affinity matrix w randomly.

        Parameters
        ----------
        hyperEdges: array of length E, containing the sets of hyper-edges (as tuples).

        Returns
        -------
        w0: initial affinity matrix.
        """
        w0 = self.prng.random_sample((self.D - 1, self.K))
        if hyperEdges is not None:
            ds = np.array(
                list(
                    set(np.arange(self.D - 1)).difference(
                        set([len(e) - 2 for e in hyperEdges])
                    )
                )
            )
            if len(ds) > 0:
                _log("setting certain d in w to zero:", ds)
                w0[ds] = 0.0
        return w0

    def _add_noise_input(self, par0: np.array) -> np.array:
        """Add noise to input values."""
        max_entry = np.max(par0)
        return par0 + max_entry * self.noise_input_par * self.prng.random_sample(
            par0.shape
        )

    def _initial_update_u_psi(self, r: int):
        """First update of the matrices u, psiOmega and psiBarOmega."""
        self.u = np.copy(self.u0_dummy)
        self.u_old = np.copy(self.u0_dummy)

        for i in range(self.N):
            self._update_psiBarOmega(i)

            if i not in self.isolates:
                self.u[i] = self.u0_current_real_t0[i]
                self.u[i] /= self.u[i].sum()
                low_values_indices = (
                    self.u[i] < self.min_value_par
                )  # values are too low
                self.u[i][low_values_indices] = 0.0  # and set to 0.
            else:
                self.u[i] = np.zeros(self.K)

            self._update_psiOmega(i=i, r=r)

            self.u_old[i] = np.copy(self.u[i])

    def _update_psiBarOmega(self, i: int, ks: Optional[int] = None) -> bool:
        """Update psiBarOmega matrix.

        Parameters
        ----------
        i: row index.
        ks: column index.

        Returns
        -------
        success: flag to check whether the matrix psiBarOmega has all non-negative entries.
        """
        success = True
        if ks is None:
            self.psiBarOmega[0] = self.psiOmega[0] - self.u[i]
            for d in range(1, self.D):
                self.psiBarOmega[d] = (
                    self.psiOmega[d] - self.u[i] * self.psiBarOmega[d - 1]
                )
        else:
            self.psiBarOmega[0][ks] = self.psiOmega[0][ks] - self.u[i][ks]
            for d in range(1, self.D):
                self.psiBarOmega[d][ks] = (
                    self.psiOmega[d][ks] - self.u[i][ks] * self.psiBarOmega[d - 1][ks]
                )

        tmpMask = self.psiBarOmega < 0
        if np.sum(tmpMask) > 0:
            if (abs(self.psiBarOmega[tmpMask]) < 1e-3).all():
                self.psiBarOmega[tmpMask] = np.zeros_like(self.psiBarOmega[tmpMask])
        tmpMask = self.psiBarOmega < 0
        if np.sum(tmpMask) > 0:
            success = False
        return success

    def _update_psiOmega(
        self, i: int, r: Optional[int] = None, ks: Optional[int] = None
    ) -> None:
        """Update psiOmega matrix.

        Parameters
        ----------
        i: row index.
        r: number of realization.
        ks: column index.
        """
        if ks is None:
            self.psiOmega[0] = self.psiOmega[0] + (self.u[i] - self.u_old[i])
            assert np.allclose(self.psiOmega[0], self.u.sum(axis=0))
            for d in range(1, self.D):
                self.psiOmega[d] = (
                    self.psiOmega[d]
                    + (self.u[i] - self.u_old[i]) * self.psiBarOmega[d - 1]
                )
        else:
            self.psiOmega[0][ks] = self.psiOmega[0][ks] + (
                self.u[i][ks] - self.u_old[i][ks]
            )
            assert np.allclose(self.psiOmega[0], self.u.sum(axis=0))
            for d in range(1, self.D):
                self.psiOmega[d][ks] = (
                    self.psiOmega[d][ks]
                    + (self.u[i][ks] - self.u_old[i][ks]) * self.psiBarOmega[d - 1][ks]
                )

        tmpMask = self.psiOmega[d] < 0
        if np.sum(tmpMask) > 0:
            if (abs(self.psiOmega[d][tmpMask]) < 1e-3).all():
                self.psiOmega[d][tmpMask] = np.zeros_like(self.psiOmega[d][tmpMask])

        # Perform a check.
        if r == 0:
            if np.sum(self.psiOmega < 0) > 0:
                tmpMask = self.psiOmega < 0
                tmpMask2 = np.logical_and(tmpMask, abs(self.psiOmega) < 1e-3)
                self.psiOmega[tmpMask2] = abs(self.psiOmega[tmpMask2])
                tmpMask = self.psiOmega < 0
                if tmpMask.sum() > 0:
                    _log("psiOmega", self.psiOmega[tmpMask])
                    _log(np.where(tmpMask))
                _log("i=", i, self.hye_per_node[i])

    def _update_rho(self) -> None:
        """Update the rho matrix that represents the variational distribution used in the EM routine."""
        self.rho = self.w[self.HyeId2D - 2] * np.exp(
            self.binary_incidence.T @ (np.log(self.u + DEFAULT_EPS))
        )
        row_sums = np.sum(self.rho, axis=1)
        self.rho[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]

    def _update_em(self) -> None:
        """Expectation-Maximization routine.
        If the parameters are considered fixed, they will not be updated when calling the .fit method.
        """
        if not self.fix_w:
            self._update_w()
            self._update_rho()
        if not self.fix_communities:
            self._update_u()
            self._update_rho()

    def _update_w(self) -> None:
        """Update affinity matrix w."""
        for d in range(self.D - 1):
            self.w[d] = np.einsum(
                "I,Ik->k", self.hye_weights[self.HyD2eId[d]], self.rho[self.HyD2eId[d]]
            )
            Z = self.gammaW + self.psiOmega[d + 1]
            non_zeros = Z > 0
            self.w[d, non_zeros] /= Z[non_zeros]
        # dist_w = np.amax(abs(self.w - self.w_old))
        self.w_old = np.copy(self.w)

    def _update_u(self) -> None:
        """Update membership matrix u. It is parallel for all the k of a node, but sequential in nodes (with random
        permutation of the nodes).
        """
        perm = self.prng.permutation(range(self.N))
        for i in perm:
            ks = np.where(self.u[i] > self.min_value_par)[0]
            if ks.shape[0] != 0:
                success = self._update_psiBarOmega(i, ks=ks)
                if success:
                    u_tmp = np.einsum(
                        "I,Ik->k",
                        self.incidence[[i], self.hye_per_node[i]],
                        self.rho[self.hye_per_node[i]][:, ks],
                    )
                    u_tmp_den = self.gammaU + np.sum(
                        self.w[:, ks] * self.psiBarOmega[:-1, ks], axis=0
                    )  # sum over d

                    if self.normalizeU:
                        low_values_indices = (u_tmp / u_tmp_den) < self.min_value_par
                        u_tmp[low_values_indices] = 0

                        lambda_i = self.enforce_constraint_u(u_tmp, u_tmp_den)
                        self.u[i, ks] = u_tmp / (lambda_i + u_tmp_den)
                    else:
                        self.u[i, ks] = u_tmp / u_tmp_den

                    self.check_u(i, ks)

                    low_values_indices = (
                        self.u[i] < self.min_value_par
                    )  # values are too low
                    self.u[i][low_values_indices] = 0  # and set to 0.

                    high_values_indices = (
                        self.u[i] > self.max_value_par
                    )  # values are too high
                    self.u[i][high_values_indices] = 1e2  # and set to 100.

                    self._update_psiOmega(i=i, ks=ks)
            self.u_old[i] = np.copy(self.u[i])
            # dist_u = np.amax(abs(self.u - self.u_old))

    @staticmethod
    def enforce_constraint_u(num: np.array, den: float) -> float:
        """Return the lagrangian multiplier to enforce the constraint on the matrix u.

        Parameters
        ----------
        num: numerator of the update of the membership matrix u.
        den: denominator of the update of the membership matrix u.

        Returns
        -------
        lambda_i: lagrangian multiplier.
        """
        lambda_i_test = root(
            func_lagrange_multiplier, x0=np.array([0.1]), args=(num, den)
        )
        lambda_i = lambda_i_test.x
        return lambda_i

    def check_u(self, i: int, ks: Optional[int]) -> None:
        """Check the updated value of u[i]."""
        tmpMask = self.u[i, ks] < 0
        if np.sum(tmpMask) > 0:
            if abs(self.u[i, ks][tmpMask]).any() > 1e-01:
                self.u[i, ks] = abs(self.u[i, ks])
            else:
                raise Warning("WARNING!", i, self.u[i])

    def _check_for_convergence(
        self, it: int, loglik: float, n_tolerance_reached: int, converged: bool
    ) -> Tuple[float, int, bool]:
        """Check for convergence by using the log-likelihood.

        Parameters
        ----------
        it: iteration.
        loglik: log-likelihood value.
        n_tolerance_reached: number of time the update of the log-likelihood respects the tolerance.
        converged: flag for reached convergence.

        Returns
        -------
        loglik: log-likelihood value.
        n_tolerance_reached: number of time the update of the log-likelihood respects the tolerance.
        converged: flag for reached convergence.
        """
        if not it % self.check_convergence_every:
            old_L = loglik
            loglik = self._LogLikelihood()
            if abs(loglik - old_L) < self.tolerance:
                n_tolerance_reached += 1
            else:
                n_tolerance_reached = 0
        if n_tolerance_reached > self.threshold_for_convergence:
            converged = True
        return loglik, n_tolerance_reached, converged

    def _LogLikelihood(self, EPS: int = 1e-300) -> float:
        """Compute the log-likelihood of the data.

        Parameters
        ----------
        EPS: random noise.

        Returns
        -------
        loglik: log-likelihood value.
        """
        # Priors.
        loglik = -self.gammaW * np.sum(self.w) - self.gammaU * np.sum(self.u)

        tmp = self.w[self.HyeId2D - 2] * np.exp(
            self.binary_incidence.T @ (np.log(self.u + DEFAULT_EPS))
        )
        loglik += np.sum(self.hye_weights * np.log(np.sum(tmp, axis=1) + EPS))

        loglik -= np.sum(
            self.w[np.arange(self.D - 1)] * self.psiOmega[np.arange(1, self.D)]
        )

        if np.isnan(loglik):
            _log("Log-likelihood is NaN!!")
            return -DEFAULT_INF
        else:
            return loglik

    def _update_optimal_parameters(self) -> None:
        """Update values of the parameters after convergence."""
        self.u_f = np.copy(self.u)
        self.w_f = np.copy(self.w)

    def _output_results(self, it_of_convergence: Optional[int]) -> None:
        """Save the results in a .npz file.

        Parameters
        ----------
        it_of_convergence: iteration of convergence.
        """
        # TODO: check if we should add this function somewhere else, i.e., in the readwrite.
        outfile = self.out_folder + "theta" + self.end_file
        np.savez_compressed(
            outfile + ".npz",
            u=self.u_f,
            w=self.w_f,
            final_it=it_of_convergence,
            maxL=self.maxL,
            non_isolates=self.non_isolates,
        )
        _log(f'Inferred parameters saved in: {outfile + ".npz"}')
        _log('To load: theta=np.load(filename), then e.g. theta["u"]')


def u0_w0_from_nparray(input_array: np.array) -> np.array:
    """Import an array."""
    return input_array


def u0_w0_from_file(filename: str, par: str) -> np.array:
    """Load an array from a .npz file.

    Parameters
    ----------
    filename: path of the stored file.
    par: name of the parameter to import.

    Returns
    -------
    Array to use as initialization for the given parameter.
    """
    theta = np.load(filename)
    return theta[par]


def calculate_u_HySC(hypergraph: Hypergraph, K: int, seed: int) -> np.array:
    """Calculate the memberships with the Hypergraph Spectral Clustering.

    Parameters
    ----------
    hypergraph: the hypergraph to perform inference on.
    K:number of communities.
    seed: random seed.

    Returns
    -------
    Membership matrix.
    """
    sc_model = HySC(seed=seed)
    return sc_model.fit(hypergraph, K=K)


def func_lagrange_multiplier(lambda_i: float, num: np.array, den: float) -> float:
    """Return the objective function to find the lagrangian multiplier to enforce the constraint on the matrix u."""
    f = num / (lambda_i + den)
    return np.sum(f) - 1


logger = logging.getLogger(__name__)


def _log(*args, **kwargs):
    message = " ".join(str(a) for a in args)
    logger.info(message)
