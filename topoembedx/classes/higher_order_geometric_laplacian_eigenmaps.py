import networkx as nx
import numpy as np
from karateclub import GLEE
from toponetx.classes import (
    CellComplex,
    CombinatorialComplex,
    DynamicCombinatorialComplex,
    SimplicialComplex,
)


def _neighbohood_from_complex(
    cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}
):
    """

    Returns the indices and matrix for the neighborhood specified by `neighborhood_type`
    and `neighborhood_dim` for the input complex `cmplex`.

    Parameters
    ----------
    cmplex : SimplicialComplex or CellComplex or CombinatorialComplex or DynamicCombinatorialComplex
        The complex to compute the neighborhood for.
    neighborhood_type : str
        The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
    neighborhood_dim : dict
        The dimensions of the neighborhood to use. If `neighborhood_type` is "adj", the dimension is
        `neighborhood_dim['r']`. If `neighborhood_type` is "coadj", the dimension is `neighborhood_dim['k']`
        and `neighborhood_dim['r']` specifies the dimension of the ambient space.

        Note:
            here neighborhood_dim={"r": 1, "k": -1} specifies the dimension for
            which the cell embeddings are going to be computed.
            r=1 means that the embeddings will be computed for the first dimension.
            The integer 'k' is ignored and only considered
            when the input complex is a combinatorial complex.

    Returns
    -------
    ind : list
        A list of the indices for the nodes in the neighborhood.
    A : ndarray
        The matrix representing the neighborhood.

    Raises
    ------
    ValueError
        If the input `cmplex` is not a SimplicialComplex, CellComplex, CombinatorialComplex, or
        DynamicCombinatorialComplex.
    """

    if isinstance(cmplex, SimplicialComplex) or isinstance(cmplex, CellComplex):
        if neighborhood_type == "adj":
            ind, A = cmplex.adjacency_matrix(neighborhood_dim["r"], index=True)

        else:
            ind, A = cmplex.coadjacency_matrix(neighborhood_dim["r"], index=True)
    elif isinstance(cmplex, CombinatorialComplex) or isinstance(
        cmplex, DynamicCombinatorialComplex
    ):
        if neighborhood_type == "adj":
            ind, A = cmplex.adjacency_matrix(
                neighborhood_dim["r"], neighborhood_dim["k"], index=True
            )
        else:
            ind, A = cmplex.coadjacency_matrix(
                neighborhood_dim["k"], neighborhood_dim["r"], index=True
            )
    else:
        ValueError(
            "input cmplex must be SimplicialComplex,CellComplex,CombinatorialComplex, or DynamicCombinatorialComplex "
        )

    return ind, A


class HOGLEE(GLEE):
    def __init__(self, dimensions: int = 3, seed: int = 42):

        super().__init__(dimensions=dimensions, seed=seed)

        self.A = []
        self.ind = []

    def fit(self, cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):
        """
        Fitting a Higher Order Geometric Laplacian EigenMaps model.

        Arg types:
            * **cmplex** *(TopoNetx Complex)* - The Complex to be embedded.
        """
        self.ind, self.A = _neighbohood_from_complex(
            cmplex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(HOGLEE, self).fit(g)

    def get_embedding(self, get_dic=False):
        emb = super(HOGLEE, self).get_embedding()
        if get_dic:
            return dict(zip(self.ind, emb))
        else:
            return emb
