"""ComplexHeat class for heat-diffusion embeddings on topological domains."""

from collections.abc import Hashable
from typing import Literal

import numpy as np
import toponetx as tnx
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD

from topoembedx.neighborhood import neighborhood_from_complex


class ComplexHeat:
    """Approximate heat-diffusion embedding algorithm for topological domains.

    Parameters
    ----------
    dimensions : int, default=128
        Final embedding dimension.
    diffusion_times : list[float], default=[0.1, 1.0, 10.0]
        Diffusion times used to build heat-kernel signatures.
    approximation_rank : int or None, default=None
        Number of Laplacian eigenpairs used in the heat-kernel approximation.
        If None, a rank is chosen automatically from the requested embedding
        dimension.
    seed : int, default=42
        Random seed value used for dimensionality reduction.
    """

    A: csr_matrix
    ind: list[Hashable]
    _embedding: np.ndarray

    def __init__(
        self,
        dimensions: int = 128,
        diffusion_times: list[float] | None = None,
        approximation_rank: int | None = None,
        seed: int = 42,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be a positive integer.")

        if diffusion_times is None:
            diffusion_times = [0.1, 1.0, 10.0]

        if not diffusion_times:
            raise ValueError(
                "`diffusion_times` must contain at least one positive value."
            )
        if any(time <= 0 for time in diffusion_times):
            raise ValueError("`diffusion_times` must contain only positive values.")
        if approximation_rank is not None and approximation_rank < 1:
            raise ValueError("approximation_rank must be a positive integer.")

        self.dimensions = dimensions
        self.diffusion_times = diffusion_times.copy()
        self.approximation_rank = approximation_rank
        self.seed = seed

    @staticmethod
    def _normalized_laplacian(A: csr_matrix) -> csr_matrix:
        """Build the normalized Laplacian from a sparse neighborhood matrix.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            Sparse neighborhood matrix of the complex.

        Returns
        -------
        scipy.sparse.csr_matrix
            Symmetric normalized Laplacian matrix.
        """
        adjacency = A.astype(np.float64).tocsr()
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()

        inv_sqrt_degree = np.zeros_like(degrees, dtype=np.float64)
        nonzero = degrees > 0
        inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

        degree_scaling = diags(inv_sqrt_degree)
        normalized_adjacency = degree_scaling @ adjacency @ degree_scaling
        diagonal = np.ones(adjacency.shape[0], dtype=np.float64)
        diagonal[~nonzero] = 0.0
        laplacian = diags(diagonal, format="csr", dtype=np.float64)
        laplacian = laplacian - normalized_adjacency

        return ((laplacian + laplacian.T) * 0.5).tocsr()

    def _resolved_approximation_rank(self, number_of_nodes: int) -> int:
        """Choose the spectral rank used by the approximation.

        Parameters
        ----------
        number_of_nodes : int
            Number of cells represented in the neighborhood graph.

        Returns
        -------
        int
            Spectral rank used by the approximation routine.
        """
        max_rank = max(number_of_nodes - 1, 1)
        if self.approximation_rank is not None:
            return min(self.approximation_rank, max_rank)
        return min(max(self.dimensions, 32), max_rank)

    def _single_node_signatures(self) -> np.ndarray:
        """Compute exact signatures for the single-node edge case.

        Returns
        -------
        numpy.ndarray
            Diffusion signatures for the single-node case.
        """
        return np.ones((1, len(self.diffusion_times)), dtype=np.float64)

    def _approximate_heat_signatures(self, laplacian: csr_matrix) -> np.ndarray:
        """Compute compact heat signatures from a truncated spectral basis.

        Parameters
        ----------
        laplacian : scipy.sparse.csr_matrix
            Sparse normalized Laplacian matrix of the neighborhood graph.

        Returns
        -------
        numpy.ndarray
            Concatenated heat signatures derived from the truncated spectral basis.
        """
        number_of_nodes = laplacian.shape[0]
        if number_of_nodes == 1:
            return self._single_node_signatures()

        rank = self._resolved_approximation_rank(number_of_nodes)
        eigenvalues, eigenvectors = eigsh(laplacian, k=rank, which="SM")
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        features = []

        for time in self.diffusion_times:
            diffusion = np.exp(-time * eigenvalues)
            features.append(eigenvectors * diffusion[np.newaxis, :])

        return np.concatenate(features, axis=1)

    def _resize_embedding(self, features: np.ndarray) -> np.ndarray:
        """Resize diffusion signatures to the requested output width.

        Parameters
        ----------
        features : numpy.ndarray
            Diffusion signature matrix before output resizing.

        Returns
        -------
        numpy.ndarray
            Embedding matrix resized to the requested output dimension.
        """
        if features.shape[1] > self.dimensions:
            reducer = TruncatedSVD(n_components=self.dimensions, random_state=self.seed)
            return reducer.fit_transform(features)

        if features.shape[1] < self.dimensions:
            padding = np.zeros(
                (features.shape[0], self.dimensions - features.shape[1]),
                dtype=features.dtype,
            )
            return np.hstack((features, padding))

        return features

    def fit(
        self,
        domain: tnx.Complex,
        neighborhood_type: Literal["adj", "coadj"] = "adj",
        neighborhood_dim=None,
    ) -> None:
        """Fit a ComplexHeat model.

        Parameters
        ----------
        domain : toponetx.classes.Complex
            A complex object. The complex object can be one of the following:
            - CellComplex
            - CombinatorialComplex
            - PathComplex
            - SimplicialComplex
            - ColoredHyperGraph
        neighborhood_type : {"adj", "coadj"}, default="adj"
            The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
        neighborhood_dim : dict
            The integer parameters needed to specify the neighborhood of the cells to
            generate the embedding. In TopoNetX (co)adjacency neighborhood matrices are
            specified via one or two parameters.

        Notes
        -----
        Here neighborhood_dim={"rank": 1, "via_rank": -1} specifies the dimension for
        which the cell embeddings are going to be computed.
        "rank": 1 means that the embeddings will be computed for the first dimension.
        The integer "via_rank": -1 is ignored when the input is cell/simplicial complex
        and must be specified when the input complex is a combinatorial complex or
        colored hypergraph.
        """
        self.ind, self.A = neighborhood_from_complex(
            domain, neighborhood_type, neighborhood_dim
        )
        laplacian = self._normalized_laplacian(self.A)
        features = self._approximate_heat_signatures(laplacian)
        self._embedding = self._resize_embedding(features)

    def get_embedding(self, get_dict: bool = False) -> dict | np.ndarray:
        """Get embedding.

        Parameters
        ----------
        get_dict : bool, optional
            Whether to return a dictionary. Defaults to False.

        Returns
        -------
        dict or numpy.ndarray
            Embedding.
        """
        if get_dict:
            return dict(zip(self.ind, self._embedding, strict=True))
        return self._embedding
