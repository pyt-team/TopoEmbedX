"""Higher Order Laplacian Eigenmaps."""

from typing import Literal

import networkx as nx
import numpy as np
import toponetx as tnx
from karateclub import LaplacianEigenmaps
from scipy.sparse import csr_matrix

from topoembedx.neighborhood import neighborhood_from_complex


class HigherOrderLaplacianEigenmaps(LaplacianEigenmaps):
    """Class for Higher Order Laplacian Eigenmaps.

    Parameters
    ----------
    dimensions : int, default=3
        Dimensionality of embedding.
    maximum_number_of_iterations : int, default=100
        Maximum number of iterations.
    seed : int, default=42
        Random seed value.
    """

    A: csr_matrix
    ind: list

    def __init__(
        self,
        dimensions: int = 3,
        maximum_number_of_iterations: int = 100,
        seed: int = 42,
    ):
        super().__init__(dimensions=dimensions, seed=seed)
        self.maximum_number_of_iterations = maximum_number_of_iterations

    def fit(
        self,
        complex: tnx.Complex,
        neighborhood_type: Literal["adj", "coadj"] = "adj",
        neighborhood_dim=None,
    ) -> None:
        """Fit a Higher Order Laplacian Eigenmaps model.

        Parameters
        ----------
        complex : toponetx.classes.Complex
            A complex object. The complex object can be one of the following:
            - CellComplex
            - CombinatorialComplex
            - PathComplex
            - SimplicialComplex
            - ColoredHyperGraph
        neighborhood_type : {"adj", "coadj"}, default="adj"
            The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
        neighborhood_dim : dict
            The integer parmaters needed to specify the neighborhood of the cells to generate the embedding.
            In TopoNetX  (co)adjacency neighborhood matrices are specified via one or two parameters.
            - For Cell/Simplicial/Path complexes (co)adjacency matrix is specified by a single parameter, this is precisely
            neighborhood_dim["rank"].
            - For Combinatorial/ColoredHyperGraph the (co)adjacency matrix is specified by a single parameter, this is precisely
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
            complex, neighborhood_type, neighborhood_dim
        )

        self.A.setdiag(1)
        g = nx.from_numpy_array(self.A)

        super().fit(g)

    def get_embedding(self, get_dict: bool = False) -> dict | np.ndarray:
        """Get embeddings.

        Parameters
        ----------
        get_dict : bool, default=False
            Whether to return a dictionary of the embedding.

        Returns
        -------
        dict or np.ndarray
            The embedding of the complex.
        """
        emb = super().get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb, strict=True))
        return emb
