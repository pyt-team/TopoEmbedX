"""Cell2Vec: a class that extends the Node2Vec class."""

from typing import Literal

import networkx as nx
import numpy as np
import toponetx as tnx
from karateclub import Node2Vec
from scipy.sparse import csr_matrix

from topoembedx.neighborhood import neighborhood_from_complex


class Cell2Vec(Node2Vec):
    """Class Cell2Vec.

    Cell2Vec is a class that extends the Node2Vec class.
    It provides additional functionality for generating node embeddings for simplicial,
    cell, combinatorial, or dynamic combinatorial complexes. The Cell2Vec class takes
    as input a simplicial, cell, combinatorial, or dynamic combinatorial complex, and
    uses the adjacency matrix or coadjacency matrix of the complex to create a graph
    object using the networkx library. The Cell2Vec class then uses this graph object
    to generate node embeddings using the Node2Vec algorithm. The Cell2Vec class allows
    users to specify the type of adjacency or coadjacency matrix to use for the graph
    (e.g. "adj" for adjacency matrix or "coadj" for coadjacency matrix), as well as the
    dimensions of the neighborhood to use for the matrix (e.g. the "adj" and "coadj"
    values for the matrix).

    Parameters
    ----------
    walk_number : int, default=10
        Number of random walks to start at each node.
    walk_length : int, default=80
        Length of random walks.
    p : float, default=1.0
        Return parameter (1/p transition probability) to move towards from previous node.
    q : float, default=1.0
        In-out parameter (1/q transition probability) to move away from previous node.
    dimensions : int, default=128
        Dimensionality of embedding.
    workers : int, default=4
        Number of cores.
    window_size : int, default=5
        Matrix power order.
    epochs : int, default=1
        Number of epochs.
    use_hierarchical_softmax : bool, default=True
        Whether to use hierarchical softmax or negative sampling to train the model.
    number_of_negative_samples : int, default=5
        Number of negative nodes to sample (usually between 5-20). If set to 0, no negative sampling is used.
    learning_rate : float, default=0.05
        HogWild! learning rate.
    min_count : int, optional
        Minimal count of node occurrences.
    seed : int, default=42
        Random seed value.
    """

    A: csr_matrix
    ind: list

    def __init__(
        self,
        walk_number: int = 10,
        walk_length: int = 80,
        p: float = 1.0,
        q: float = 1.0,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        use_hierarchical_softmax: bool = True,
        number_of_negative_samples: int = 5,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(
            walk_number,
            walk_length,
            p,
            q,
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
        """Fit a Cell2Vec model.

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

        g = nx.from_numpy_array(self.A)
        g.add_edges_from((index, index) for index in range(g.number_of_nodes()))

        super().fit(g)

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
        emb = super().get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb, strict=True))
        return emb
