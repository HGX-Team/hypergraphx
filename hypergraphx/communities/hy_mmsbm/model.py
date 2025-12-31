from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse

from hypergraphx import Hypergraph
from hypergraphx.communities.hy_mmsbm._linear_ops import bf, bf_and_sum, qf, qf_and_sum
from hypergraphx.linalg.linalg import binary_incidence_matrix


class HyMMSBM:
    """Implementation of the Hy-MMSBM probabilistic model from

    "Community Detection in Large Hypergraphs",
    Ruggeri N., Contisciani M., Battiston F., De Bacco C., and

    "A framework to generate hypergraphs with community structure"
    Ruggeri N., Battiston F., De Bacco C.

    The probabilistic model assumes the formation of hyperedges according to a Poisson
    distribution. The Poisson distribution for every single hyperedge is determined by a
    common affinity matrix w and soft-community assignments u for the nodes.
    """

    def __init__(
        self,
        K: Optional[int] = None,
        u: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        assortative: Optional[bool] = None,
        kappa_fn: str = "binom+avg",
        max_hye_size: Optional[int] = None,
        u_prior: Union[float, np.ndarray] = 0.0,
        w_prior: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize the probabilistic model.
        The parameters u and w can be provided as input, either both, only one or none.
        The parameters provided at initialization are considered fixed, and will not be
        inferred when calling the .fit method.

        Parameters
        ----------
        K: number of communities.
            The input is optional if either the affinity matrix w or the assignments u
            are provided.
        u: community soft assignments.
            This is a matrix of shape (N, K), with N number of nodes in the hypergraph.
            Every row i contains the soft assignments for node i.
        w: affinity matrix.
            The affinity matrix is symmetric with shape (K, K).
        assortative: whether the affinity matrix w is expected to be diagonal.
            This input is optional if w is provided.
        kappa_fn: form of the kappa normalization for the hyperedges' probabilities.
            For now, only the choice utilized in the reference paper has been
            implemented.
        max_hye_size: maximum hyperedge size.
            This parameter is utilized for computing quantities relative to the model,
            for example the expected degree, but is otherwise not enforced.
            For example, during EM-inference, it is up to the user to provide as input a
            hypergraph respecting this parameter, no check will be executed.
        u_prior: rate for the exponential prior on u.
            It can be provided as a float, in which case it specifies the same prior
            parameter for all the entries of u, or as an array of (possible different)
            separate exponential rates for every entry of u. If it is an array, it needs
            to have same shape as u.
            To avoid specifying a prior for u, set u_prior to the 0. float value.
        w_prior: rate for the exponential prior on u.
            Similar to the exponential rate for u. If an array, it needs to be a
            symmetric matrix with same shape as w.
            To avoid specifying a prior for w, set w_prior to the 0. float value.
        seed: random seed.
        """
        super().__init__()

        # Number of communities.
        self.K = K
        # Whether to use a diagonal or full affinity matrix.
        self.assortative = assortative

        # Node soft assignments.
        self.u = u
        # Affinity matrix.
        self.w = w

        # Exponential prior parameters.
        self.u_prior = u_prior
        self.w_prior = w_prior

        self._check_and_infer_param_consistency()

        # Other model properties.
        self.kappa_fn = kappa_fn
        self.max_hye_size = max_hye_size

        # Training related attributes.
        self.tolerance: Optional[float] = None  # Tolerance for EM stopping criterion.
        self.trained: bool = False  # The model has been trained or not.
        self.training_iter: Optional[int] = None  # Number of EM iterations performed.
        self.tolerance_reached: bool = False  # Stopping criterion satisfied.

        # Random number generator.
        self._rng: np.random.Generator = np.random.default_rng(seed)

    @property
    def N(self) -> Union[None, int]:
        """Total number of nodes in the hypergraph."""
        if self.u is None:
            return None
        return self.u.shape[0]

    def C(
        self, d: Union[str, int, np.ndarray] = "all", return_summands: bool = False
    ) -> Union[np.ndarray, float]:
        """Constant for calculation of likelihood. It has formula

        .. math::
            \sum_{d=2}^D \binom{N-2}{d-2}/ kappa_d

        Parameters
        ----------
        d: single value or array of values for the hyperedge dimension.
        return_summands: since C consists of a sum of terms, specify if to only return
            the final sum or the summands separately.

        Returns
        -------
        The C value, or its summands, according to return_summands.
        """
        d_vals = self._dimensions_to_numpy(d)

        if self.kappa_fn == "binom+avg":
            res = 2 / (d_vals * (d_vals - 1))
        else:
            raise NotImplementedError()

        if not return_summands:
            res = res.sum()

        return res

    def fit(
        self,
        hypergraph: Hypergraph,
        n_iter: int = 500,
        tolerance: Optional[float] = None,
        check_convergence_every: int = 10,
    ) -> None:
        """Perform Expectation-Maximization inference on a hypergraph, as presented  in

        "Community Detection in Large Hypergraphs",
        Ruggeri N., Contisciani M., Battiston F., De Bacco C.,

        The inference can be performed both on the affinity matrix w and the assignments
        u.
        If either or both have been provided as input at initialization of the model,
        they are regarded as ground-truth and are not inferred.

        Parameters
        ----------
        hypergraph: the hypergraph to perform inference on.
        n_iter: maximum number of EM iterations.
        tolerance: tolerance for the stopping criterion.
        check_convergence_every: number of steps in between every convergence check.
        """
        # Initialize all the values needed for training.
        self.tolerance = tolerance
        self.tolerance_reached = False

        if self.w is None:
            fixed_w = False
            self._init_w()
        else:
            fixed_w = True

        if self.u is None:
            fixed_u = False
            self._init_u(hypergraph.num_nodes())
        else:
            fixed_u = True

        # Infer the maximum hyperedge size if not already specified inside the model.
        max_hye_size_data = max(len(hye) for hye in hypergraph)
        if self.max_hye_size is None:
            self.max_hye_size = max_hye_size_data
        else:
            if self.max_hye_size < max_hye_size_data:
                raise ValueError(
                    "The hypergraph contains hyperedges with size greater than that "
                    "specified in the model. This will not influence training, but "
                    "might cause other modeling problems. If you want max_hye_size to "
                    "be detected automatically, set it to None."
                )

        binary_incidence = binary_incidence_matrix(hypergraph)
        hye_weights = np.array(hypergraph.get_weights())

        # Train.
        for it in range(n_iter):
            if not fixed_w:
                self.w = self._w_update(binary_incidence, hye_weights)
            if not fixed_u:
                self.u = self._u_update(binary_incidence, hye_weights)

            # Check for convergence.
            if tolerance is not None:
                if (not it % check_convergence_every) and (it > 0):
                    converged = (
                        np.linalg.norm(self.w - old_w) / self.K < tolerance
                        and np.linalg.norm(self.u - old_u) / hypergraph.num_nodes()
                        < tolerance
                    )
                    if converged:
                        self.tolerance_reached = True
                        break
                old_w, old_u = self.w, self.u
        # As pointed out in the paper, the C constant can be absorbed in one of the
        # parameter sets during inference, and simply be accounted for at the end of the
        # optimization procedure.
        if not fixed_w:
            self.w = self.w / self.C()
        elif not fixed_u:
            self.u = self.u / np.sqrt(self.C())

        self.trained = True
        self.training_iter = it

    def poisson_params(
        self,
        binary_incidence: Union[np.ndarray, sparse.spmatrix],
        return_edge_sum: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the Poisson parameters for all the hyperedges.
        Given the binary incidence matrix, of shape (N, E), return the parameters for
        the edges in a tensor of shape (E,).
        Notice that the parameters returned are not the final Poisson means for the
        hyperedges, as they are not normalized by the kappa constants.
        Formally, this function returns the values defined as lambda_e in the paper.
        For every hyperedge e, they are defined as

        .. math::
            \sum_{i < j \in e} u_i^T w u_j

        Parameters
        ----------
        binary_incidence: the binary incidence matrix
        return_edge_sum: whether to return the edge sums.
            These quantities are computed as an intermediate value.
            For a single hyperedge e they are defined as

            .. math::
                \sum_{i \in e} u_i

            and are collected in a vector of length E.

        Returns
        -------
        The Poisson parameters in a vector of length E. Optionally, the edge sums in
        another vector of length E.
        """
        self._check_u_w_init()
        u, w = self.u, self.w

        E = binary_incidence.shape[1]
        K = w.shape[0]

        # First addend: for every hyperedge e: s_e^T w s_e .
        edge_sum = self._edge_sum(binary_incidence)
        assert edge_sum.shape == (E, K)

        first_addend = qf(edge_sum, w)
        assert first_addend.shape == (E,)

        # Second addend: for every hyperedge e: sum_{i \in e} u_i^T w u_i .
        second_addend = binary_incidence.T @ qf(u, w)
        assert second_addend.shape == (E,)

        poisson_params = 0.5 * (first_addend - second_addend)

        if return_edge_sum:
            return poisson_params, edge_sum
        return poisson_params

    def expected_degree(
        self,
        per_node: bool = False,
        d: Union[str, int, np.ndarray] = "all",
    ) -> Union[np.ndarray, float]:
        """Compute the expected degree according to the probabilistic model.
        If per_node=True, the expected degree is computed for the single nodes, else for
        the full hypergraph. Notice that the expected degree depends on the interaction
        sizes taken into account. These can be specified manually by giving an array of
        integers d (or a single integer).

        Parameters
        ----------
        per_node: whether the expected degree needs to be computed for the single nodes,
            or averaged.
        d: interactions to take into account to compute the expected degree.
            This can be an array of integers, a single integer, or the string "all".
            Using d="all" is equivalent to
            `d=numpy.arange(2, self.max_hye_size+1)`.

        Returns
        -------
        A float with the average degree if per_node=False, else the array with the
        expected degrees of the single nodes.
        """
        if self.max_hye_size is None:
            raise ValueError(
                "Cannot compute the expected degree if "
                "no max hyperedge size has been specified."
            )
        self._check_u_w_init()
        u, w = self.u, self.w
        N, K = self.N, self.K

        if per_node:
            C = self.C(d)
            C_prime = self._C_prime(d)

            u_sum = u.sum(axis=0)
            assert u_sum.shape == (K,)

            first_addend = bf(u, u_sum, w) - qf(u, w)
            assert first_addend.shape == (N,)

            second_addend = 0.5 * (qf(u_sum - u, w) - qf_and_sum(u, w) + qf(u, w))
            assert second_addend.shape == (N,)

            return C * first_addend + C_prime * second_addend
        else:
            C_second = self._C_second(d)
            return C_second * bf_and_sum(u, w)

    def log_likelihood(
        self,
        hypergraph: Hypergraph,
    ) -> float:
        """Compute the log-likelihood of the model on a given hypergraph.

        Parameters
        ----------
        hypergraph: the hypergraph to compute the log-likelihood of.

        Returns
        -------
        The log-likelihood value.
        """
        self._check_u_w_init()
        u, w = self.u, self.w

        binary_incidence = binary_incidence_matrix(hypergraph)
        hye_weights = np.array(hypergraph.get_weights())

        # First addend: all interactions u_i * w * u_j .
        first_addend = bf_and_sum(u, w)

        # Second addend: interactions in the hypergraph A_e * log(lambda_e) .
        second_addend = np.dot(
            hye_weights, np.log(self.poisson_params(binary_incidence))
        )

        return -first_addend + second_addend

    def degree_sequence(
        self,
        include_dyadic: bool = False,
        expected: bool = False,
    ) -> np.ndarray:
        """Approximately sample the degree sequence from the model using the Central
        Limit Theorem. The degree sequence depends on the interactions to take into
        account. If include_dyadic, then also binary interactions (i.e. edges) are taken
        into account, otherwise only interactions of size three or higher are.

        Parameters
        ----------
        include_dyadic: include the order-two hyperedges (i.e. edges) or not
            in the calculation of the degree sequence.
        expected: return the analytical expected value of the degree sequence, or sample
            the sequence.

        Returns
        -------
        The N-dimensional array with the sampled degree sequence.
        """
        dims = (
            np.arange(2, self.max_hye_size + 1)
            if include_dyadic
            else np.arange(3, self.max_hye_size + 1)
        )
        mean = self.expected_degree(per_node=True, d=dims)

        if expected:
            return mean

        return sample_discretized_positive_gaussian(
            loc=mean, scale=np.sqrt(mean), rng=self._rng
        )

    def dimension_sequence(
        self,
        include_dyadic: bool = False,
        expected: bool = False,
    ) -> Dict[int, int]:
        """Approximately sample the dimension sequence from the model using the Central
        Limit Theorem. The dimension sequence is a dictionary with {key: value} pairs
        {dimension: number of hyperedges with that dimension}
        If include_dyadic, also binary interactions (i.e. edges) are sampled.

        Parameters
        ----------
        include_dyadic: whether to sample the number of order-two interactions.
        expected: return the analytical expected value of the dimension sequence, or
            sample the sequence.

        Returns
        -------
        The dictionary representing the sampled dimension sequence.
        """
        dims = (
            np.arange(2, self.max_hye_size + 1)
            if include_dyadic
            else np.arange(3, self.max_hye_size + 1)
        )
        if self.kappa_fn == "binom+avg":
            count_constant = self.C(dims, return_summands=True)
            mean = count_constant * bf_and_sum(self.u, self.w)
        else:
            raise NotImplementedError()

        if expected:
            return {
                dim: mean_count for dim, mean_count in zip(dims, mean) if mean_count > 0
            }

        hye_count = sample_discretized_positive_gaussian(
            loc=mean, scale=np.sqrt(mean), rng=self._rng
        )
        dim_seq = {dim: count for dim, count in zip(dims, hye_count) if count > 0}
        return dim_seq

    def log_kappa(self, d: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the normalization constant kappa(d) in log-space.

        Parameters
        ----------
        d: float or array of values d to compute the function kappa over.

        Returns
        -------
        The function value of log(kappa(d)).
        """
        if self.kappa_fn == "binom+avg":
            if isinstance(d, np.ndarray):
                log_binomial_ = np.vectorize(log_binomial, excluded={"n"})
            else:
                log_binomial_ = log_binomial
            return (
                log_binomial_(self.N - 2, d - 2) + np.log(d) + np.log(d - 1) - np.log(2)
            )
        else:
            raise NotImplementedError()

    def sample_dyadic_interactions(self) -> np.ndarray:
        """Sample the undirected graph with parameters from the probabilistic model.
        This is returned as a square upper-triangular matrix.
        """
        poisson_lambda = bf(self.u, self.u, self.w) / np.exp(self.log_kappa(2))
        adjacency = self._rng.poisson(poisson_lambda) > 0
        return np.triu(adjacency, 1)

    def _w_update(
        self,
        binary_incidence: Union[np.ndarray, sparse.spmatrix],
        hye_weights: np.ndarray,
    ) -> np.ndarray:
        """EM or MAP updates for the affinity matrix w."""
        u, w = self.u, self.w

        E = len(hye_weights)
        N = u.shape[0]
        K = self.K

        poisson_params, edge_sum = self.poisson_params(
            binary_incidence, return_edge_sum=True
        )

        multiplier = hye_weights / poisson_params
        assert multiplier.shape == (E,)

        # Numerator : first addend s_ea * s_eb .
        first_addend = np.matmul(edge_sum.T, edge_sum * multiplier[:, None])
        assert first_addend.shape == (K, K)

        # Numerator: second addend u_ia * u_ib .
        if sparse.issparse(binary_incidence):
            weighting = binary_incidence.multiply(multiplier[None, :]).sum(axis=1)
            weighting = np.asarray(weighting).reshape(-1)
        else:
            weighting = (binary_incidence * multiplier[None, :]).sum(axis=1)
        assert weighting.shape == (N,)
        second_addend = np.matmul(u.T, u * weighting[:, None])
        assert second_addend.shape == (K, K)

        numerator = 0.5 * w * (first_addend - second_addend)

        # Denominator.
        u_sum = u.sum(axis=0)
        assert u_sum.shape == (K,)
        denominator = 0.5 * (np.outer(u_sum, u_sum) - np.matmul(u.T, u))
        assert denominator.shape == (K, K)

        return numerator / (denominator + self.w_prior)

    def _u_update(
        self,
        binary_incidence: Union[np.ndarray, sparse.spmatrix],
        hye_weights: np.ndarray,
    ) -> np.ndarray:
        """EM or MAP updates for the community assignments u."""
        u, w = self.u, self.w

        E = len(hye_weights)
        N = u.shape[0]
        K = self.K

        # Numerator.
        poisson_params, edge_sum = self.poisson_params(
            binary_incidence, return_edge_sum=True
        )

        multiplier = hye_weights / poisson_params
        assert multiplier.shape == (E,)

        if sparse.issparse(binary_incidence):
            weighting = binary_incidence.multiply(multiplier[None, :])
            assert sparse.issparse(weighting)
        else:
            weighting = binary_incidence * multiplier[None, :]
        assert weighting.shape == (N, E)

        first_addend = weighting @ edge_sum
        assert first_addend.shape == (N, K)

        if sparse.issparse(weighting):
            weighting_sum = np.asarray(weighting.sum(axis=1)).reshape(-1, 1)
        else:
            weighting_sum = weighting.sum(axis=1, keepdims=True)
        second_addend = weighting_sum * u
        assert second_addend.shape == (N, K)

        numerator = u * np.matmul(first_addend - second_addend, w)

        # Denominator.
        u_sum = u.sum(axis=0)
        assert u_sum.shape == (K,)
        denominator = np.matmul(w, u_sum)[None, :] - np.matmul(u, w)
        assert denominator.shape == (N, K)

        return numerator / (denominator + self.u_prior)

    def _C_prime(self, d: Union[str, int, np.ndarray] = "all") -> float:
        """Only utilized for computing the expected degree of single nodes.
        It has formula:

        .. math::
            \sum_{d=3}^D \binom{N-3}{d-3}/ kappa_d
        """
        d_vals = self._dimensions_to_numpy(d)

        if self.kappa_fn == "binom+avg":
            return 2 / (self.N - 2) * np.sum((d_vals - 2) / (d_vals * (d_vals - 1)))
        else:
            raise NotImplementedError()

    def _C_second(self, d: Union[str, int, np.ndarray] = "all") -> float:
        """Only utilized for computing the expected degree.
        It has formula:

        .. math::
            \sum_{d=2}^D \binom{N-2}{d-2} d / kappa_d
        """
        d_vals = self._dimensions_to_numpy(d)

        if self.kappa_fn == "binom+avg":
            return 2 / self.N * np.sum(1 / (d_vals - 1))
        else:
            raise NotImplementedError()

    def _dimensions_to_numpy(
        self, d: Union[str, int, np.ndarray] = "all"
    ) -> np.ndarray:
        """Convenience method, take some allowed values d of the hypergedge dimensions
        and return the degree in array format.
        If d is already a numpy array, it is returned.
        If d is an integer, it is wrapped inside an array.
        If d is the string "all", then the array of all possible dimensions is returned,
        as specified by self.max_hye_size.
        """
        if isinstance(d, str) and d == "all" and self.max_hye_size is None:
            raise ValueError(
                "self.max_hye_size has not been specified. "
                "Either specify it or given d as input."
            )
        elif isinstance(d, str) and d == "all":
            d_vals = np.arange(2, self.max_hye_size + 1)
        elif isinstance(d, str):
            raise ValueError('Only string value for d is "all"')
        elif isinstance(d, int):
            d_vals = np.array([d])
        else:
            d_vals = d

        return d_vals

    def _check_and_infer_param_consistency(self) -> None:
        if self.assortative is None:
            if self.w is None:
                raise ValueError(
                    "self.assortative cannot be inferred since self.w is None. "
                    "Provide either w or assortative as input."
                )
            else:
                self.assortative = np.all(np.triu(self.w, 1) == 0)

        if self.K is None:
            if self.u is None and self.w is None:
                raise ValueError(
                    "Number of communities K cannot be inferred since self.w and self.u"
                    " are None. Provide either w, u, or K as input."
                )
            elif self.w is not None:
                self.K = self.w.shape[0]
            elif self.u is not None:
                self.K = self.u.shape[1]

        if self.w is not None:
            if np.any(self.w < 0):
                raise ValueError("The adjacency matrix w contains negative entries.")

            if not np.all(self.w == self.w.T):
                raise ValueError("The adjacency matrix w is not symmetric.")

            if self.assortative and np.any(np.triu(self.w, 1) != 0):
                raise ValueError(
                    "The model is assortative, but the "
                    "adjacency matrix w is not diagonal."
                )

        if self.u is not None:
            if np.any(self.u < 0):
                raise ValueError("The assignment matrix u contains negative entries.")

        if self.u is not None and self.w is not None:
            if not self.u.shape[1] == self.w.shape[0]:
                raise ValueError("The number of communities of u and w are different.")

    def _init_w(self) -> None:
        K = self.K
        rng = self._rng

        if isinstance(self.w_prior, float) and self.w_prior == 0.0:
            w = rng.random((K, K))
            self.w = np.triu(w, 0) + np.triu(w, 1).T
            if self.assortative:
                self.w = np.diag(np.diag(self.w))

        else:
            if self.assortative:
                if isinstance(self.w_prior, float):
                    prior_mean = np.diag(np.eye(K) / self.w_prior)
                else:
                    prior_mean = 1 / np.diag(self.w_prior)
                self.w = np.diag(rng.exponential(prior_mean))

            else:
                if isinstance(self.w_prior, float):
                    prior_mean = np.ones((K, K)) / self.w_prior
                else:
                    prior_mean = 1 / self.w_prior
                w = rng.exponential(prior_mean)
                self.w = np.triu(w, 0) + np.triu(w, 1).T

    def _init_u(self, N: int) -> None:
        K = self.K
        rng = self._rng

        if isinstance(self.u_prior, float) and self.u_prior == 0.0:
            self.u = rng.random((N, K))
        else:
            if isinstance(self.u_prior, np.ndarray):
                self.u = rng.exponential(1 / self.u_prior)
            else:
                self.u = rng.exponential(1 / self.u_prior, size=(N, K))

    def _check_u_w_init(self):
        if self.u is None or self.w is None:
            raise ValueError("The parameters w and u are not initialized.")

    def _edge_sum(
        self, binary_incidence: Union[np.ndarray, sparse.spmatrix]
    ) -> np.ndarray:
        """Return the sum of the soft-assignments u_i in the specified hyperedges.
        For a specific hyperedge e, this is the K-dimensional vector given by

        .. math::
            \sum_{i \in e} u_i

        Parameters
        ----------
        binary_incidence: sparse or full binary incidence matrix of shape (N, E),
            representing the hyperedges.
            Here, E is the number of hyperedges, which is arbitrary, while N is the
            number of nodes in the hypergraph, which needs to match the one specified by
            the (N, K) soft assignment matrix u.

        Returns
        -------
        An array of shape (E, K), containing the hyperedge sum for all the hyperedges
        given as input.
        """
        return binary_incidence.T @ self.u


def log_binomial(n: int, k: int) -> float:
    """Compute the logarithm of the binomial coefficient of n over k."""
    return np.log(np.arange(n - k + 1, n + 1)).sum() - np.log(np.arange(1, k + 1)).sum()


def sample_discretized_positive_gaussian(
    loc: np.array, scale: np.array, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Sample a Gaussian, then round its samples to the nearest integer
    and clip them to be greater than 0.
    """
    rng = rng if rng is not None else np.random.default_rng()
    samples = rng.normal(loc=loc, scale=scale)
    samples = np.rint(samples)
    samples[samples < 0] = 0.0
    return samples.astype(int)
