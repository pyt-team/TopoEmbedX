import networkx as nx
from karateclub import DeepWalk, Node2Vec
from toponetx.classes import (
    CellComplex,
    CombinatorialComplex,
    DynamicCombinatorialComplex,
    SimplicialComplex,
)


def _neighbohood_from_complex(
    cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}
):

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


class DeepCell(DeepWalk):
    """

    Parameters
    ==========

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
        """
        Parameters
        ==========
        - walk_number (int): The number of random walks to generate for each node.
        - walk_length (int): The length of each random walk.
        - p (float): Return hyperparameter for the random walks.
        - q (float): In-out hyperparameter for the random walks.
        - dimensions (int): The dimensionality of the embedding vector.
        - workers (int): The number of parallel workers to use for training.
        - window_size (int): The size of the sliding window.
        - epochs (int): The number of iterations (epochs) over the corpus.
        - learning_rate (float): The learning rate for the model.
        - min_count (int): The minimum count of words to consider when training the model.
        - seed (int): Random seed to use for reproducibility.
        """

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

    def fit(self, cmplex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):
        self.ind, self.A = _neighbohood_from_complex(
            cmplex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(DeepCell, self).fit(g)

    def get_embedding(self, get_dic=False):
        emb = super(DeepCell, self).get_embedding()
        if get_dic:
            return dict(zip(self.ind, emb))
        else:
            return emb
