"""DeepCell class for embedding complex networks using DeepWalk."""

import networkx as nx
from karateclub import DeepWalk

from topoembedx.neighborhood import neighborhood_from_complex


class DeepCell(DeepWalk):
    """Class for DeepCell.

    Parameters
    ----------
    walk_number : int, optional
        Number of random walks to generate for each node. Defaults to 10.
    walk_length : int, optional
        Length of each random walk. Defaults to 80.
    dimensions : int, optional
        Dimensionality of embedding. Defaults to 128.
    workers : int, optional
        Number of parallel workers to use for training. Defaults to 4.
    window_size : int, optional
        Size of the sliding window. Defaults to 5.
    epochs : int, optional
        Number of iterations (epochs). Defaults to 1.
    learning_rate : float, optional
        Learning rate for the model. Defaults to 0.05.
    min_count : int, optional
        Minimum count of words to consider when training the model. Defaults to 1.
    seed : int, optional
        Random seed to use for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        super().__init__(
            walk_number=walk_number,
            walk_length=walk_length,
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
        self,
        complex,
        neighborhood_type="adj",
        neighborhood_dim={"rank": 0, "via_rank": -1},
    ):
        """Fit the model.

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

        super(DeepCell, self).fit(g)

    def get_embedding(self, get_dict=False):
        """Get embeddings.

        Parameters
        ----------
        get_dict : bool, optional
            Return a dictionary of the embedding.
            Default: False.

        Returns
        -------
        dict or np.ndarray
            The embedding of the complex.
        """
        emb = super(DeepCell, self).get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb))
        return emb
