"""MCMC sampler for the Hy-MMSBM probabilistic model, described in

"A framework to generate hypergraphs with community structure"
Ruggeri N., Contisciani M., Battiston F., De Bacco C.

Notice that the sampler is separate from the probabilistic model itself.
In this view, the sampler only takes care of any sampling-related operation,
and refers back to the model to retrieve any probability-related value
(e.g. Poisson probabilities, expected degree, etc.).
"""
import logging
import warnings
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats

from hypergraphx import Hypergraph
from hypergraphx.communities.hy_mmsbm.model import HyMMSBM
from hypergraphx.linalg.linalg import hye_list_to_binary_incidence


class HyMMSBMSampler:
    """Sampler for the Hy-MMSBM model.
    This class takes care of the approximate sampling routine described in

    "A framework to generate hypergraphs with community structure"
    Ruggeri N., Contisciani M., Battiston F., De Bacco C.
    """

    def __init__(
        self,
        u: np.ndarray = None,
        w: np.ndarray = None,
        max_hye_size: Optional[int] = None,
        exact_dyadic_sampling: bool = True,
        burn_in_steps: int = 1000,
        intermediate_steps: int = 1000,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize the sampler instance.

        Parameters
        ----------
        u: community soft assignments.
            This is a matrix of shape (N, K), with N number of nodes in the hypergraph.
            Every row i contains the soft assignments for node i.
        w: affinity matrix.
            The affinity matrix is symmetric with shape (K, K).
        max_hye_size: maximum size of the hyperedges.
            If None, the maximum size is the number of nodes in the hypergraph.
        exact_dyadic_sampling: whether to sample the order-two interactions (i.e. edges)
            "exactly" from their Bernoulli distribution, or approximately via Central
            Limit Theorem.
        burn_in_steps: number of burn-in steps for Metropolis-Hastings MCMC.
        intermediate_steps: number of steps in between returned samples for
            Metropolis-Hastings MCMC.
        seed: random seed.
        """
        self.intermediate_steps = intermediate_steps
        self.burn_in_steps = burn_in_steps
        self.exact_dyadic_sampling = exact_dyadic_sampling

        self._model = HyMMSBM(
            u=u,
            w=w,
            max_hye_size=max_hye_size if max_hye_size else len(u),
        )

        # Attributes for sampling diagnostics.
        self.iter_count: int = 0
        self.accept_count: int = 0
        self.reject_count: int = 0

        self.matching_sequences: Optional[bool] = None

        # Random number generator.
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def sample(
        self,
        deg_seq: Optional[np.ndarray] = None,
        dim_seq: Optional[Dict[int, int]] = None,
        avg_deg: Optional[float] = None,
        initial_hyg: Optional[Hypergraph] = None,
        allow_rescaling: bool = False,
    ) -> Iterable[Hypergraph]:
        """Approximate hypergraph sampling routine presented in

        "A framework to generate hypergraphs with community structure"
        Ruggeri N., Contisciani M., Battiston F., De Bacco C.

        Possibly, condition the sampling on different quantities: the expected average
        degree, a given degree sequence and/or dimension sequence, or a hypergraph.

        Parameters
        ----------
        deg_seq: degree sequence
            This is specified as an array of degrees, one per node in the hypergraph.
        dim_seq: dimension sequence
            This is specified as a dictionary with {key: value} pairs
            {dimension: number of hyperedges with that dimension}
        avg_deg: average degree
        initial_hyg: an initial hypergraph to start the MCMC from.
            If initial_hyg is provided, all the other inputs (i.e. deg_seq, dim_seq,
            avg_deg, allow_rescaling) are ignored. The MCMC is started using
            initial_hyg as first configuration, which is statistically equivalent to
            conditioning on its degree and dimension sequences. During MCMC, the degree
            and dimension sequences in initial_hyg are preserved.
        allow_rescaling: allow the rescaling of the u and w parameters in place.
            This adjusts for the sampling constraints, e.g. average degree.

        Returns
        -------
        A generator of sampled hypergraphs.
        Notice that all the generated hypergraphs are conditioned on the same degree and
        dimension sequences (possibly provided as input, otherwise sampled at the
        beginning and kept constant). To sample conditioning on different sequences, a
        new call to this method is required.
        """
        if initial_hyg is None:
            samples = self._sampling_from_sequences(
                deg_seq, dim_seq, avg_deg, allow_rescaling
            )
        else:
            # Map from node to an id in [0, N) where N is the number of nodes.
            mapping = initial_hyg.get_mapping()
            initial_config = [
                set(hye) for hye in map(mapping.transform, initial_hyg.get_edges())
            ]
            samples = self._mcmc_routine(initial_config)

        while True:
            hye_list = next(samples)
            hye_list = [tuple(sorted(hye)) for hye in hye_list]
            binary_incidence = hye_list_to_binary_incidence(
                hye_list, shape=(self._model.N, len(hye_list))
            )
            hye_size = np.fromiter(map(len, hye_list), dtype=int)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_poisson = np.log(
                    self._model.poisson_params(binary_incidence)
                ) - self._model.log_kappa(hye_size)
            poisson_mean = np.exp(log_poisson)
            # Weights can't be zero, remedy numerical underflow by clipping.
            poisson_mean = np.clip(poisson_mean, a_min=1.0e-10, a_max=None)
            weights = sample_truncated_poisson(poisson_mean, self._rng).astype(int)

            # Although theoretically impossible, sometimes the sampled weights are
            # zero due to numerical instabilities.
            # Remove the relative hyperedges from the samples.
            nonzero = np.where(weights > 0)[0]
            weights = weights[nonzero]
            hye_list = [hye_list[idx] for idx in nonzero]

            # Map indices in [0, N) back to the original nodes.
            if initial_hyg:
                hye_list = [
                    tuple(edge) for edge in map(mapping.inverse_transform, hye_list)
                ]

            # Remove duplicates and sum weights
            hye_with_weights = {edge: 0 for edge in set(hye_list)}
            for edge, w in zip(hye_list, weights):
                hye_with_weights[edge] += w

            yield Hypergraph(
                edge_list=list(hye_with_weights),
                weighted=True,
                weights=list(hye_with_weights.values()),
            )

    def _sampling_from_sequences(
        self,
        deg_seq: Optional[np.ndarray] = None,
        dim_seq: Optional[Dict[int, int]] = None,
        avg_deg: Optional[float] = None,
        allow_rescaling: bool = True,
    ) -> Iterator[List[Set[int]]]:
        """Construct the MCMC chain that yields unweighted hypergraphs.
        The logic is different depending on the conditioning on various quantities.

        Possible quantities to condition on are degree sequence, dimension sequence and
        average degree.
        If either or both the degree and dimension sequence are not provided, they are
        sampled directly from the probabilistic model.
        Successively, they are arranged into a first proposal binary hypergraph for
        MCMC. To condition directly on a given hypergraph, see the _mcmc_routine method.

        Parameters
        ----------
        deg_seq: degree sequence, optional
        dim_seq: dimension sequence, optional
            The {key: value} pairs are given by
            {dimension: number of hyperedges with that dimension}
        avg_deg: average degree, optional
        allow_rescaling: whether to allow scalar rescaling of the model's parameters.
            This is needed if one or more of the two sequences or the average degree are
            provided. In such case, rescaling is performed to have the model's
            parameters yield expected statics close to the provided ones.

        Yields
        -------
        Generated samples of binary hypergraphs. The hypergraphs are represented as a
        lists of hyperedges. Hyperedges are represented as sets of nodes.
        """
        N = self._model.N

        # Adjust the model's parameters based on the constraints provided in input.
        # This is done by rescaling the parameters of the model to attain minimum
        # L2-distance between the sequences required and those expected from the model.
        # Alternatively, rescale the model to attain a desired expected degree.
        # The parameters of the model are rescaled in place.
        if allow_rescaling:
            self._rescale_model_parameters(deg_seq, dim_seq, avg_deg)
        else:
            if avg_deg is not None:
                logging.warning(
                    "Provided an expected average degree, but allow_rescaling=False."
                    "Ignoring the requirement on the average degree."
                )

        # Sample the degree and dimension sequences if not provided as input.
        # For higher-order interactions, the sampling is done via Central Limit Theorem.
        # For dyadic interactions, we sample the single edges and compute the resulting
        # degree and dimension sequences.
        sample_deg_seq = deg_seq is None
        sample_dim_seq = dim_seq is None
        if sample_deg_seq:
            deg_seq = np.zeros(N)
        if sample_dim_seq:
            dim_seq = dict()

        if (sample_deg_seq or sample_dim_seq) and self.exact_dyadic_sampling:
            edges = self._model.sample_dyadic_interactions()
            if sample_dim_seq and sample_deg_seq:
                # If none between the degree sequence and the dimension sequence are
                # provided, simply sample the dyadic interactions and pass them
                # alongside the other hyperedges found after MCMC.
                # This helps in solving issues with repeated hyperedges during mixing of
                # the chain, since these happen mainly at the dyadic interaction level.
                pass
            elif sample_deg_seq:
                deg_seq += edges.sum(axis=1)
            elif sample_dim_seq:
                dim_seq[2] = edges.sum()

        if sample_deg_seq:
            deg_seq += self._model.degree_sequence(
                include_dyadic=not self.exact_dyadic_sampling
            )

        if sample_dim_seq:
            dim_seq = {
                **dim_seq,
                **self._model.dimension_sequence(
                    include_dyadic=not self.exact_dyadic_sampling
                ),
            }

        assert deg_seq.shape == (N,)
        assert all(dim <= N for dim in dim_seq), (
            "The dimension sequence contains values that exceed the number of nodes "
            "in the hypergraph."
        )

        # Both the dimension sequence and the degree sequence determine the sum
        # of the degrees in the hypergraph.
        # The sequences need to match for sampling to be mathematically possible.
        # If they don't, we perturb them. This is done in a constructive way by building
        # the hyperedge list and dynamically modifying the sequences when needed.
        hye_list = self._match_sequences(
            deg_seq,
            dim_seq,
            force_deg_seq=not sample_deg_seq,
            force_dim_seq=not sample_dim_seq,
        )
        if sample_deg_seq and sample_dim_seq and self.exact_dyadic_sampling:
            # Order-two hyperedges are kept fixed during MCMC.
            fixed_hyperedges = [set(hye) for hye in zip(*np.nonzero(edges))]
        else:
            fixed_hyperedges = None

        # Once we have the initial configuration, start the MCMC procedure.
        return self._mcmc_routine(hye_list, fixed_hyperedges=fixed_hyperedges)

    def _mcmc_routine(
        self,
        hye_list: List[Set[int]],
        fixed_hyperedges: Optional[List[Set[int]]] = None,
    ) -> Iterator[List[Set[int]]]:
        """MCMC routine, for conditional sampling of a binary hypergraph given an
        initial configuration.
        The initial configuration is a list of hyperedges, which are represented as sets
        of nodes. Optionally, a list of fixed hyperedges can be provided. These are not
        mixed during the MCMC procedure, but are added as-is to the returned samples.

        Parameters
        ----------
        hye_list: list of hyperedges for the initial configuration.
            hye_list is a list of hyperedges, represented as sets of nodes.
        fixed_hyperedges: a fixed list of hyperedges.
            These hyperedges are not mixed within the Markov chain, but merged with the
            samples form the chain to form the hypergraph yield at every iteration.

        Yields
        -------
        Generated samples of binary hypergraphs. The hypergraphs are represented as a
        lists of hyperedges. Hyperedges are represented as sets of nodes.
        """
        if fixed_hyperedges is None:
            fixed_hyperedges = []

        for _ in range(self.burn_in_steps):
            self._mcmc_step(hye_list)

        while True:
            for _ in range(self.intermediate_steps):
                self._mcmc_step(hye_list)
            yield hye_list.copy() + fixed_hyperedges.copy()
            self.iter_count += 1

    def _mcmc_step(self, hye_list: List[Set[int]]) -> None:
        """Perform one MCMC step consisting of shuffling and accept-reject.
        Modify the input list of hyperedges in place.
        """
        # Select two random hyperedges.
        idx1, idx2 = self._rng.choice(len(hye_list), size=2, replace=False)
        hye1, hye2 = hye_list[idx1], hye_list[idx2]

        # Reshuffle hyperedges.
        new_hye1, new_hye2 = self._pairwise_reshuffle(hye1, hye2)

        # Get Poisson parameters for the four hyperedges.
        hye1 = tuple(hye1)
        hye2 = tuple(hye2)
        new_hye1 = tuple(new_hye1)
        new_hye2 = tuple(new_hye2)

        incidence_matrix = hye_list_to_binary_incidence(
            [hye1, hye2, new_hye1, new_hye2], shape=(self._model.N, 4)
        )
        poisson_lambda = self._model.poisson_params(incidence_matrix)
        log_kappa = self._model.log_kappa(
            np.array([len(hye1), len(hye2), len(new_hye1), len(new_hye2)])
        )

        # Transition probability and accept-reject step.
        transition_prob = self._transition_prob(poisson_lambda, log_kappa)
        if self._rng.random() < transition_prob:
            hye_list[idx1] = set(new_hye1)
            hye_list[idx2] = set(new_hye2)
            self.accept_count += 1
        else:
            self.reject_count += 1

    def _pairwise_reshuffle(
        self, hye1: Set[int], hye2: Set[int]
    ) -> Tuple[Set[int], Set[int]]:
        """Given two hyperedges, perform the random reshuffling operation proposed in
        "Configuration Models of Random Hypergraphs", Chodrow 2020.

        Parameters
        ----------
        hye1: the first hyperedge to be shuffled, represented as a set of nodes
        hye2: the second hyperedge to be shuffled, represented as a set of nodes

        Returns
        -------
        The two new hyperedges new_hye1 and new_hye2, as sets of nodes.
        Furthermore, new_hye1 has same dimension as hye1, new_hye2 the same as hye2.
        """
        intersection = hye1 & hye2
        disjoint_union = (hye1 | hye2) - intersection

        new_hye1 = set(
            self._rng.choice(
                list(disjoint_union), size=len(hye1) - len(intersection), replace=False
            )
        )
        new_hye2 = disjoint_union - new_hye1

        new_hye1 = new_hye1 | intersection
        new_hye2 = new_hye2 | intersection

        assert len(new_hye1) == len(hye1)
        assert len(new_hye2) == len(hye2)

        return new_hye1, new_hye2

    def _rescale_model_parameters(
        self,
        deg_seq: Optional[np.ndarray] = None,
        dim_seq: Optional[Dict[int, int]] = None,
        avg_deg: Optional[float] = None,
    ) -> None:
        """Rescale the model's parameters w, u, based on the input constraints.
        In general, the rescaling takes the form of a multiplication by a constant
        of either w or u. The constant is chosen to reduce the difference between
        the input constraints and the values expected from the model.

        Parameters
        ----------
        deg_seq: degree sequences.
        dim_seq: dimension sequence.
        avg_deg: average degree.
            This requirement is ignored if one between deg_seq or dim_seq is provided.
        """
        if deg_seq is not None or dim_seq is not None:
            if avg_deg is not None:
                logging.warning(
                    "Sampling conditioned on both expected degree and dimension or "
                    "degree sequences has been required."
                    "Due to mismatches, the average degree requirement will be ignored."
                )
            expected_values = np.zeros(0)
            input_values = np.zeros(0)

            if deg_seq is not None:
                input_values = np.concatenate([input_values, deg_seq])
                expected_values = np.concatenate(
                    [
                        expected_values,
                        self._model.degree_sequence(include_dyadic=True, expected=True),
                    ]
                )

            if dim_seq is not None:
                expected_dim_seq = self._model.dimension_sequence(
                    include_dyadic=True, expected=True
                )
                all_dims = list(set(dim_seq.keys()) | set(expected_dim_seq.keys()))

                input_values = np.concatenate(
                    [input_values, np.array([dim_seq.get(dim, 0) for dim in all_dims])]
                )
                expected_values = np.concatenate(
                    [
                        expected_values,
                        np.array([expected_dim_seq.get(dim, 0) for dim in all_dims]),
                    ]
                )

            norm = np.linalg.norm(expected_values)
            if norm != 0.0:
                rescaling_const = np.inner(input_values, expected_values) / norm**2
                self._model.u *= np.sqrt(rescaling_const)
        elif avg_deg is not None:
            avg_deg_model = self._model.expected_degree(per_node=False, d="all")
            rescaling_const = avg_deg / avg_deg_model
            self._model.u *= np.sqrt(rescaling_const)

    @staticmethod
    def _transition_prob(
        poisson_lambda: np.ndarray, log_kappa: np.ndarray, approx_thresh=5
    ) -> float:
        """Compute the transition probability ratio for the Metropolis-Hastings
        algorithm on the shuffled hyperedges.
        The Poisson means for the four hyperedges are passed as input together with the
        log of the kappa normalization from the binomial form model.

        Parameters
        ----------
        poisson_lambda: array with Poisson means of the four hyperedges.
            The following order is assumed for the four values:
            (old_hyperedge1, old_hyperedge2, new_hyperedge1, new_hyperedge2).
        log_kappa: logarithm of the kappa normalization values.
            The same order as for the poisson_lambda values is assumed.
        approx_thresh: numerical threshold to apply numerical approximations.

        Returns
        -------
        The probability ratio.
        """
        assert log_kappa[0] == log_kappa[2] and log_kappa[1] == log_kappa[3]

        # To avoid numerical errors with parameters underflowing to 0.
        poisson_lambda = poisson_lambda + 1.0e-30
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_lambda = np.log(poisson_lambda)

        prob_ratio = 1.0
        for old, new in [(0, 2), (1, 3)]:
            if np.all((log_kappa - log_lambda)[[old, new]] > approx_thresh):
                prob_ratio *= poisson_lambda[new] / poisson_lambda[old]
            else:
                prob_old = np.exp(log_lambda[old] - log_kappa[old])
                prob_new = np.exp(log_lambda[new] - log_kappa[new])
                prob_ratio *= np.clip(
                    np.expm1(prob_new) / np.expm1(prob_old),
                    a_min=1.0e-30,
                    a_max=1.0e30,
                )
        return prob_ratio

    def _match_sequences(
        self,
        deg_seq: np.ndarray,
        dim_seq: Dict[int, int],
        force_deg_seq: bool = False,
        force_dim_seq: bool = True,
    ) -> List[Set[int]]:
        """Given the degree and dimension sequence, match them while building a
        hyperedge list.
        Both the dimension sequence and the degree sequence determine the sum of the
        degrees in the hypergraph. These need to match for sampling to be mathematically
        possible. If they don't, the sequences need to be perturbed.
        The match is performed in a constructive way by sampling the hyperedges and
        dynamically modifying the sequences if needed.
        The function allows which sequence to keep intact in case of mismatches.

        Parameters
        ----------
        deg_seq: degree sequence
            This is specified as an array of degrees, one per node in the hypergraph.
        dim_seq: dimension sequence
            This is specified as a dictionary with {key: value} pairs
            {size: number of hyperedges with that size}
        force_deg_seq: whether to keep the degree sequence intact in case of mismatches
            with the dimension sequence.
            If also force_dim_seq is set to True, this value is ignored.
        force_dim_seq: whether to keep the dimension sequence intact in case of
            mismatches with the degree sequence.
            if also force_deg_seq is set to True, this value has precedence in the case
            of mismatches.

        Returns
        -------
        The list of hyperedges built based on the given degree and dimension sequences.
        """
        nodes_with_deg = self._deg_seq_to_dict(deg_seq)
        hye_list = []
        # Exhaust hyperedge size requirement, up to conflicts between degree and
        # dimension sequences.
        for hye_size in dim_seq:  # Hyperedge size.
            for _ in range(dim_seq[hye_size]):  # Number of hyperedges with given size.
                new_hye = self._extract_hye(
                    nodes_with_deg, hye_size, force_deg_seq, force_dim_seq
                )
                if len(new_hye) > 1:
                    hye_list.append(new_hye)

        # Exhaust node degree requirements, up to conflicts between degree and
        # dimension sequences.
        if force_deg_seq and not force_dim_seq:
            if any(deg != 0 for deg in nodes_with_deg):
                self.matching_sequences = False
                available_nodes = sum(
                    len(node_set) for deg, node_set in nodes_with_deg.items() if deg > 0
                )
                while available_nodes > 1:
                    hye_size = self._rng.integers(2, self.model.max_hye_size + 1)
                    new_hye = self._extract_hye(
                        nodes_with_deg, hye_size, force_deg_seq, force_dim_seq
                    )
                    assert len(new_hye) > 1
                    hye_list.append(new_hye)
                    available_nodes = sum(
                        len(node_set)
                        for deg, node_set in nodes_with_deg.items()
                        if deg > 0
                    )

        if self.matching_sequences is None:
            self.matching_sequences = True

        return hye_list

    @staticmethod
    def _deg_seq_to_dict(deg_seq: np.ndarray) -> Dict[int, Set[int]]:
        """Take a degree sequence in array form, and return an equivalent dictionary
        with {key:value} pairs given by {degree: set of nodes with that degree}.
        """
        nodes_with_deg = dict()
        for node, deg in enumerate(deg_seq):
            nodes_with_deg[deg] = nodes_with_deg.get(deg, set()) | {
                node,
            }
        return nodes_with_deg

    def _extract_hye(
        self,
        nodes_with_deg: Dict[int, Set[int]],
        hye_size: int,
        force_deg_seq: bool = False,
        force_dim_seq: bool = True,
    ) -> Set[int]:
        """The main algorithmic step in inferring a hyperedge list from some degree and
        dimension sequence.
        This step involves taking all the nodes available to generate a hyperedge with
        the specified size.
        The nodes are chosen in order starting from the ones with the highest degree.

        All the nodes are contained in the dictionary nodes_with_deg, which has
        {key:value} pairs specified as {degree:set of nodes with this degree}.

        In some cases, building the desired hyperedge can be impossible due to
        mismatches between the nodes available and the required hyperedge size.
        If force_deg_seq is True, then a smaller (possibly empty) hyperedge is returned,
        and the nodes' degrees are preserved.
        If force_dim_seq is True, then some nodes with degree zero could be added to the
        hyperedge to reach the desired size.
        If both are set to True, force_dim_seq has precedence.
        If both are set to False, the effect is the same as for force_dim_seq=True.

        After the hyperedge is built, the input dictionary nodes_with_deg is modified in
        place: the degree of the nodes selected is lowered by 1.
        """
        if hye_size < 1:
            raise ValueError(f"Invalid hye_size: {hye_size}")

        nodes_chosen = dict()
        n_nodes_sampled = 0
        degrees = iter(
            sorted((deg for deg in nodes_with_deg.keys() if deg > 0), reverse=True)
        )
        while n_nodes_sampled < hye_size:
            try:
                deg = next(degrees)
            except StopIteration:
                logging.info(
                    "There aren't anymore nodes available to choose."
                    "Breaking the condition on either the dimension or degree sequence."
                )
                self.matching_sequences = False
                # If force_dim_seq, add nodes with degree 0 to the hyperedge.
                if force_dim_seq or not force_deg_seq:
                    if force_dim_seq and force_deg_seq:
                        logging.warning(
                            f"{self.__class__.__name__}: both force_deg_seq and"
                            "force_dim_seq have been set to True, causing conflicts. "
                            "Ignoring the constraints on the degree sequence."
                        )
                    nodes_chosen[0] = set(
                        self._rng.choice(
                            list(nodes_with_deg[0]),
                            size=hye_size - n_nodes_sampled,
                            replace=False,
                        )
                    )
                    n_nodes_sampled = hye_size
                    continue
                # If force_deg_seq, reduce the hyperedge size to the number of nodes
                # available. Avoid returning singletons, in such case return empty
                # hyperedges.
                else:
                    if n_nodes_sampled == 1:
                        nodes_chosen = {0: set()}
                    break

            n_nodes_to_sample = min(
                len(nodes_with_deg[deg]), hye_size - n_nodes_sampled
            )

            nodes_chosen[deg] = set(
                self._rng.choice(
                    list(nodes_with_deg[deg]),
                    size=n_nodes_to_sample,
                    replace=False,
                )
            )
            n_nodes_sampled += n_nodes_to_sample

        for deg, node_set in nodes_chosen.items():
            if deg > 0:
                nodes_with_deg[deg] = nodes_with_deg[deg] - node_set
                nodes_with_deg[deg - 1] = nodes_with_deg.get(deg - 1, set()) | node_set

        nodes_chosen = set.union(*(node_set for node_set in nodes_chosen.values()))
        return nodes_chosen


def sample_truncated_poisson(
    lambd: Union[float, int, np.ndarray], rng: Optional[np.random.Generator] = None
) -> Union[float, np.ndarray]:
    """Sample a truncated Poisson.
    If X is a Poisson random variable with parameter lambda, the relative
    truncated Poisson variable Y with same parameter lambda is defined as
    Y = X | X > 0.
    """
    rng = rng if rng is not None else np.random.default_rng()
    u = rng.random(1) if not isinstance(lambd, np.ndarray) else rng.random(*lambd.shape)
    p = u + (1 - u) * np.exp(-lambd)
    return stats.poisson.ppf(p, lambd)
