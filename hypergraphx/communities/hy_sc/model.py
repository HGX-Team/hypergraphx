from typing import Optional, Tuple

import logging
import numpy as np

try:
    from sklearn.cluster import KMeans  # type: ignore
except ImportError:  # pragma: no cover
    KMeans = None

from hypergraphx import Hypergraph
from hypergraphx.linalg.linalg import binary_incidence_matrix, incidence_matrix


class HySC:
    """Implementation of Hypergraph Spectral Clustering from
    "Learning with hypergraphs: Clustering, classification, and embedding",
    Zhou D., Huang J., Schölkopf B.
    """

    def __init__(
        self,
        seed: int = 0,
        inf: int = 1e40,
        n_realizations: int = 10,
        out_inference: bool = False,
        out_folder: str = "../data/output/",
        end_file: str = "_sc.dat",
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        seed: random seed.
        inf: initial value for the log-likelihood.
        n_realizations: number of realizations with different random initialization.
        out_inference: flag to store the inferred parameters.
        out_folder: path to store the output.
        end_file: output file suffix.
        """
        # Training related attributes.
        self.seed = seed
        self.inf = inf
        self.n_realizations = n_realizations
        # Output related attributes.
        self.out_inference = out_inference
        self.out_folder = out_folder
        self.end_file = end_file
        # Membership matrix
        self.u: Optional[np.array] = None

    def fit(self, hypergraph: Hypergraph, K: int, weighted_L: bool = False) -> np.array:
        """Perform community detection on hypergraphs with spectral clustering.

        Parameters
        ----------
        hypergraph: the hypergraph to perform inference on.
        K: number of communities.
        weighted_L: flag to use the weighted Laplacian.

        Returns
        -------
        u: hard-membership matrix of dimension (N, K).
        """
        # Initialize all the parameters needed for training.
        self._init_data(hypergraph=hypergraph, K=K)

        # Get the Laplacian matrix.
        self._extract_laplacian(weighted_L=weighted_L)
        # Get eigenvalues and eigenvectors of the Laplacian matrix.
        e_vals, e_vecs = self.extract_eigenvectors()
        # Extract hard-memberships by applying the K-Means algorithm to the eigenvectors.
        self.u = self.apply_kmeans(e_vecs.real, seed=self.seed)

        # Save inferred parameter.
        if self.out_inference:
            self.output_results()

        return self.u

    def _init_data(
        self,
        hypergraph: Hypergraph,
        K: int,
    ):
        """Initialize parameters for the inference.

        Parameters
        ----------
        hypergraph: the hypergraph to perform inference on.
        K: number of communities.
        """
        # Weighted incidence matrix.
        self.incidence = incidence_matrix(hypergraph)
        # Binary incidence matrix.
        self.binary_incidence = binary_incidence_matrix(hypergraph)

        # Number of nodes, and number of hyperedges.
        self.N, self.E = self.incidence.shape
        # Number of communities.
        self.K = K
        # Maximum observed hyperedge size.
        self.D = max([len(e) for e in np.array(hypergraph.get_edges(), dtype=object)])

        # Nodes' degree.
        # TODO: is there a better way to get the array with the node degrees?
        self.node_degree = self.binary_incidence.sum(axis=1)
        # TODO: is there a better way to get the weighted degrees?
        self.node_degree_weighted = self.incidence.sum(axis=1)
        # Hyperedges' size.
        self.hye_size = np.array(hypergraph.get_sizes())
        # TODO: is there a better way to get the weighted sizes?
        self.hye_size_weighted = self.incidence.sum(axis=1)

        node_order = list(hypergraph.get_mapping().classes_)
        self.isolates = hypergraph.isolates(node_order=node_order)
        self.non_isolates = hypergraph.non_isolates(node_order=node_order)

    def _extract_laplacian(self, weighted_L: bool = False) -> None:
        # Check for division by zero and handle isolated nodes
        invDE = np.diag(1.0 / np.where(self.hye_size == 0, 1, self.hye_size))
        invDV2 = np.diag(
            np.sqrt(1.0 / np.where(self.node_degree == 0, 1, self.node_degree))
        )

        # set to zero for isolated nodes - we check directly on self.node_degree
        invDV2[self.node_degree == 0, self.node_degree == 0] = 0

        if weighted_L:
            dense_incidence = self.incidence.toarray()
            HT = dense_incidence.T
            self.L = np.eye(self.N) - invDV2 @ dense_incidence @ invDE @ HT @ invDV2
        else:
            dense_binary_incidence = self.binary_incidence.toarray()
            HT = dense_binary_incidence.T
            self.L = (
                np.eye(self.N) - invDV2 @ dense_binary_incidence @ invDE @ HT @ invDV2
            )

    def extract_eigenvectors(self) -> Tuple[np.array, np.array]:
        """Extract eigenvalues and eigenvectors of the Laplacian matrix."""
        e_vals, e_vecs = np.linalg.eig(self.L[self.non_isolates][:, self.non_isolates])
        sorted_indices = np.argsort(e_vals)
        return e_vals[sorted_indices[: self.K]], e_vecs[:, sorted_indices[1 : self.K]]

    def apply_kmeans(self, X: np.array, seed: int = 10) -> np.array:
        """Apply K-means algorithm to the eigenvectors of the Laplacian matrix.

        Parameters
        ----------
        X: matrix with eigenvectors.
        seed: random seed.

        Returns
        -------
        X_pred: membership matrix.
        """
        if KMeans is None:
            raise ImportError(
                "HySC requires scikit-learn. Install hypergraphx with the appropriate extra "
                "(e.g. `pip install hypergraphx[ml]`) or install scikit-learn manually."
            )
        y_pred = KMeans(
            n_clusters=self.K, random_state=seed, n_init=self.n_realizations
        ).fit_predict(X)
        X_pred = np.zeros((self.N, self.K))
        for idx, i in enumerate(self.non_isolates):
            X_pred[i, y_pred[idx]] = 1
        return X_pred

    def output_results(self) -> None:
        """Save the results in a .npz file."""
        outfile = self.out_folder + "theta" + self.end_file
        np.savez_compressed(outfile + ".npz", u=self.u)
        logger = logging.getLogger(__name__)
        logger.info("Inferred parameters saved in: %s", outfile + ".npz")
        logger.info('To load: theta=np.load(filename), then e.g. theta["u"]')
