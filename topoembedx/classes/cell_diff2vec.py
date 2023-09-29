"""Class CellDiff2Vec."""

import networkx as nx
from karateclub import Diff2Vec

from topoembedx.neighborhood import neighborhood_from_complex


class CellDiff2Vec(Diff2Vec):
    """Class for CellDiff2Vec.

    Parameters
    ----------
    diffusion_number : int, optional
        Number of diffusion. Defaults to 10.
    diffusion_cover : int, optional
        Number of nodes in diffusion. Defaults to 80.
    dimensions : int, optional
        Dimensionality of embedding. Defaults to 128.
    workers : int, optional
        Number of cores. Defaults to 4.
    window_size : int, optional
        Matrix power order. Defaults to 5.
    epochs : int, optional
        Number of epochs. Defaults to 1.
    learning_rate : float, optional
        HogWild! learning rate. Defaults to 0.05.
    min_count : int, optional
        Minimal count of node occurrences. Defaults to 1.
    seed : int, optional
        Random seed value. Defaults to 42.
    """

    def __init__(
        self,
        diffusion_number: int = 10,
        diffusion_cover: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        super().__init__(
            diffusion_number=diffusion_number,
            diffusion_cover=diffusion_cover,
            dimensions=dimensions,
            workers=workers,
            window_size=window_size,
            epochs=epochs,
            learning_rate=learning_rate,
            min_count=min_count,
            seed=seed,
        )

        self.A = []
        self.ind = []

    def fit(
        self, complex, neighborhood_type="adj", neighborhood_dim={"adj": 0, "coadj": -1}
    ):
        """Fit a CellDiff2Vec model.

        Parameters
        ----------
        complex : TopoNetX object
            A complex object. The complex object can be one of the following:
            - CellComplex
            - CombinatorialComplex
            - PathComplex
            - SimplicialComplex
            - ColoredHyperGraph
        neighborhood_type : str
            The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
        neighborhood_dim : dict
            The integer parmaters needed to specify the neighborhood of the cells to generate the embedding.
            In TopoNetX  (co)adjacency neighborhood matrices are specified via one or two parameters.
            - For Cell/Simplicial/Path complexes (co)adjacency matrix is specified by a single parameter, this is precisely
            neighborhood_dim["rank"]
            - For Combinatorial/ColoredHyperGraph the (co)adjacency matrix is specified by a single parameter, this is precisely
            neighborhood_dim["rank"] and neighborhood_dim["via_rank"]

        Notes
        -----
            Here neighborhood_dim={"rank": 1, "via_rank": -1} specifies the dimension for
            which the cell embeddings are going to be computed.
            "rank": 1 means that the embeddings will be computed for the first dimension.
            The integer "via_rank": -1 is ignored when the input is cell/simplicial complex
            and  must be specified when the input complex is a combinatorial complex or
            colored hypergraph.

        Returns
        -------
        None
        """
        self.ind, self.A = neighborhood_from_complex(
            complex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(CellDiff2Vec, self).fit(g)

    def get_embedding(self, get_dict=False):
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
        emb = super(CellDiff2Vec, self).get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb))
        return emb
