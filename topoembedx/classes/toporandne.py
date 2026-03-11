"""Topological version of the RandNE embedding algorithm."""

from collections.abc import Hashable
from typing import Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp
import toponetx as tnx
from karateclub import RandNE

from topoembedx.neighborhood import neighborhood_from_complex


class TopoRandNE(RandNE):
    """Topological version of the RandNE [1] embedding algorithm.

    Parameters
    ----------
    dimensions : int, default=128
        Number of embedding dimension.
    alphas : list[float], default=[0.5, 0.5]
        Smoothing weights for adjacency matrix powers.
    seed : int, default=42
        Random seed.

    References
    ----------
    .. [1] Zhang, Ziwei, et al. "Billion-Scale Network Embedding with Iterative Random
           Projection". 2018 IEEE International Conference on Data Mining (ICDM)
           [Singapore], 2018, pp. 787-96. https://doi.org/10.1109/ICDM.2018.00094.
    """

    A: sp.csr_matrix
    ind: list[Hashable]
    _embedding: np.ndarray

    def __init__(
        self, dimensions: int = 128, alphas: list[float] | None = None, seed: int = 42
    ) -> None:
        if alphas is None:
            alphas = [0.5, 0.5]
        super().__init__(dimensions, alphas, seed)

    def fit(
        self,
        domain: tnx.Complex,
        neighborhood_type: Literal["adj", "coadj"] = "adj",
        neighborhood_dim=None,
    ) -> None:
        """Fit the model.

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
            The integer parameters needed to specify the neighborhood of the cells to generate the embedding.
            In TopoNetX  (co)adjacency neighborhood matrices are specified via one or two parameters.
            - For Cell/Simplicial/Path complexes (co)adjacency matrix is specified by a single parameter, this is precisely
            neighborhood_dim["rank"].
            - For Combinatorial/ColoredHyperGraph the (co)adjacency matrix is specified by two parameters, this is precisely
            neighborhood_dim["rank"] and neighborhood_dim["via_rank"].

        Notes
        -----
        Here neighborhood_dim={"rank": 1, "via_rank": -1} specifies the dimension for
        which the cell embeddings are going to be computed.
        "rank": 1 means that the embeddings will be computed for the first dimension.
        The integer "via_rank": -1 is ignored when the input is cell/simplicial complex
        and  must be specified when the input complex is a combinatorial complex or
        colored hypergraph.
        """
        self.ind, self.A = neighborhood_from_complex(
            domain, neighborhood_type, neighborhood_dim
        )
        self.A.setdiag(1)

        g = nx.from_scipy_sparse_array(self.A)

        super().fit(g)

    def get_embedding(self, get_dict: bool = False) -> dict | np.ndarray:
        """Get embeddings.

        Parameters
        ----------
        get_dict : bool, default=False
            Return a dictionary of the embedding.

        Returns
        -------
        dict or np.ndarray
            The embedding of the complex.
        """
        emb = super().get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb, strict=True))
        return emb
