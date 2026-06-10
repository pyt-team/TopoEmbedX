"""Cell2Vec: a class that extends the Node2Vec class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from karateclub import Node2Vec

from topoembedx.neighborhood import neighborhood_from_complex

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import toponetx as tnx
    from scipy.sparse import csr_matrix

    from topoembedx.neighborhood import NeighborhoodType


class Cell2Vec(Node2Vec):
    """Topological version of the Node2Vec embedding algorithm.

    Cell2Vec extends Node2Vec to topological domains by first computing a
    neighborhood matrix from a complex and then applying Node2Vec to the graph
    induced by that matrix.

    Parameters
    ----------
    walk_number : int, default=10
        Number of random walks to start at each node.
    walk_length : int, default=80
        Length of random walks.
    p : float, default=1.0
        Return parameter.
    q : float, default=1.0
        In-out parameter.
    dimensions : int, default=128
        Dimensionality of embedding.
    workers : int, default=4
        Number of workers.
    window_size : int, default=5
        Window size.
    epochs : int, default=1
        Number of epochs.
    use_hierarchical_softmax : bool, default=True
        Whether to use hierarchical softmax.
    number_of_negative_samples : int, default=5
        Number of negative samples.
    learning_rate : float, default=0.05
        Learning rate.
    min_count : int, default=1
        Minimal count of node occurrences.
    seed : int, default=42
        Random seed.
    """

    A: csr_matrix
    ind: list[Hashable]

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
        neighborhood_type: NeighborhoodType = "adj",
        neighborhood_dim: Mapping[str, Any] | None = None,
    ) -> None:
        """Fit a Cell2Vec model.

        Parameters
        ----------
        domain : toponetx.classes.Complex
            A complex object.
        neighborhood_type : str, default="adj"
            The neighborhood type used to construct the graph. Supported values
            are those accepted by
            :func:`topoembedx.neighborhood.neighborhood_from_complex`, including
            ``"adj"``, ``"coadj"``, ``"connection"``, ``"hasse"``, and
            ``"augmented_hasse"``.
        neighborhood_dim : mapping, optional
            Parameters specifying the neighborhood matrix.
        """
        self.ind, self.A = neighborhood_from_complex(
            domain,
            neighborhood_type,
            neighborhood_dim,
        )
        graph = self._graph_from_adjacency(self.A)
        super().fit(graph)

    @staticmethod
    def _graph_from_adjacency(matrix: csr_matrix) -> nx.Graph:
        """Create an unweighted NetworkX graph from an adjacency matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            Sparse adjacency matrix used to construct the graph.

        Returns
        -------
        networkx.Graph
            Unweighted graph induced by the nonzero entries of ``matrix``, with
            self-loops added to all nodes.

        Notes
        -----
        KarateClub's biased random walker expects unweighted graphs to expose
        edges as ``(u, v)`` pairs. NetworkX creates weighted edges by default
        when ``edge_attr`` is not ``None``. Using ``edge_attr=None`` preserves
        the previous unweighted Cell2Vec behavior and avoids weighted-walker
        errors.
        """
        graph = nx.from_numpy_array(
            np.asarray(matrix.toarray()),
            edge_attr=None,
        )
        graph.add_edges_from(
            (index, index) for index in range(graph.number_of_nodes())
        )
        return graph

    def get_embedding(
        self,
        get_dict: bool = False,
    ) -> dict[Hashable, np.ndarray] | np.ndarray:
        """Get embedding.

        Parameters
        ----------
        get_dict : bool, default=False
            Whether to return a dictionary indexed by cell identifiers.

        Returns
        -------
        dict or numpy.ndarray
            The learned cell embeddings.
        """
        emb = super().get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb, strict=True))
        return emb
