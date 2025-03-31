"""DeepCell class for embedding complex networks using DeepWalk."""

from typing import Literal

import networkx as nx
import numpy as np
import toponetx as tnx
from karateclub import DeepWalk
from scipy.sparse import csr_matrix

from topoembedx.neighborhood import neighborhood_from_complex


class DeepCell(DeepWalk):
    """Class for DeepCell.

    Parameters
    ----------
    walk_number : int, default=10
        Number of random walks to generate for each node.
    walk_length : int, default=80
        Length of each random walk.
    dimensions : int, default=128
        Dimensionality of embedding.
    workers : int, default=4
        Number of parallel workers to use for training.
    window_size : int, default=5
        Size of the sliding window.
    epochs : int, default=1
        Number of iterations (epochs).
    use_hierarchical_softmax : bool, default=True
        Whether to use hierarchical softmax or negative sampling for training.
    number_of_negative_samples : int, default=5
        Number of negative samples to use for negative sampling.
    learning_rate : float, default=0.05
        Learning rate for the model.
    min_count : int, optional
        Minimum count of words to consider when training the model.
    seed : int, default=42
        Random seed to use for reproducibility.
    """

    A: csr_matrix
    ind: list

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        use_hierarchical_softmax: bool = True,
        number_of_negative_samples: int = 5,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        super().__init__(
            walk_number,
            walk_length,
            dimensions,
            workers,
            window_size,
            epochs,
            use_hierarchical_softmax,
            number_of_negative_samples,
            learning_rate,
            min_count,
            seed,
        )

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
            domain, neighborhood_type, neighborhood_dim
        )

        self.A.setdiag(1)
        g = nx.from_numpy_array(self.A)

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
