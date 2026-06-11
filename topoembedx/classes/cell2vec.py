"""Cell2Vec: a class that extends the Node2Vec class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from karateclub import Node2Vec

from topoembedx.neighborhood import neighborhood_from_complex

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Hashable, Mapping  # pragma: no cover

    import toponetx as tnx  # pragma: no cover
    from scipy.sparse import csr_matrix  # pragma: no cover

    from topoembedx.neighborhood import NeighborhoodType  # pragma: no cover


class Cell2Vec(Node2Vec):
    """Topological version of the Node2Vec embedding algorithm.

    Cell2Vec extends Node2Vec to topological domains by first computing a
    neighborhood matrix from a complex and then applying Node2Vec to the graph
    induced by that matrix.

    The model can use any square neighborhood graph returned by
    :func:`topoembedx.neighborhood.neighborhood_from_complex`, including
    same-rank adjacency, coadjacency, incidence-based connection graphs,
    Hasse graphs, and augmented Hasse graphs.

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

    Examples
    --------
    Fit Cell2Vec on the vertex adjacency graph of a cell complex:

    >>> import toponetx as tnx
    >>> from topoembedx.classes.cell2vec import Cell2Vec
    >>> domain = tnx.CellComplex([[0, 1, 2]])
    >>> model = Cell2Vec(
    ...     dimensions=2,
    ...     walk_number=2,
    ...     walk_length=4,
    ...     window_size=2,
    ...     workers=1,
    ... )
    >>> model.fit(
    ...     domain,
    ...     neighborhood_type="adj",
    ...     neighborhood_dim={"rank": 0},
    ... )
    >>> model.get_embedding().shape
    (3, 2)

    Fit Cell2Vec on a cross-rank connection graph between vertices and edges:

    >>> model = Cell2Vec(
    ...     dimensions=2,
    ...     walk_number=2,
    ...     walk_length=4,
    ...     window_size=2,
    ...     workers=1,
    ... )
    >>> model.fit(
    ...     domain,
    ...     neighborhood_type="connection",
    ...     neighborhood_dim={"rank": 0, "to_rank": 1},
    ... )
    >>> model.get_embedding().shape
    (6, 2)

    Fit Cell2Vec on a Hasse graph over ranks 0, 1, and 2:

    >>> model = Cell2Vec(
    ...     dimensions=2,
    ...     walk_number=2,
    ...     walk_length=4,
    ...     window_size=2,
    ...     workers=1,
    ... )
    >>> model.fit(
    ...     domain,
    ...     neighborhood_type="hasse",
    ...     neighborhood_dim={"ranks": [0, 1, 2]},
    ... )
    >>> model.get_embedding().shape
    (7, 2)
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
        neighborhood_type : {"adj", "coadj", "inc", "incidence", "connection", "hasse", "augmented_hasse"}, default="adj"
            The neighborhood type used to construct the graph. The values are
            passed directly to
            :func:`topoembedx.neighborhood.neighborhood_from_complex`.

            ``"adj"``
                Fits Cell2Vec on a same-rank adjacency graph.

            ``"coadj"``
                Fits Cell2Vec on a same-rank coadjacency graph.

            ``"inc"``, ``"incidence"``, ``"connection"``
                Fits Cell2Vec on a square graph induced by one or more
                cross-rank incidence matrices.

            ``"hasse"``
                Fits Cell2Vec on a Hasse graph over selected ranks.

            ``"augmented_hasse"``
                Fits Cell2Vec on a Hasse graph augmented with additional
                same-rank or cross-rank neighborhoods.
        neighborhood_dim : mapping, optional
            Parameters specifying the neighborhood matrix. Typical keys include
            ``"rank"``, ``"via_rank"``, ``"to_rank"``, ``"target_rank"``,
            ``"rank_pairs"``, ``"pairs"``, ``"ranks"``,
            ``"neighborhoods"``, ``"symmetric"``, and ``"ranked_labels"``.

        Examples
        --------
        Fit on a same-rank adjacency neighborhood:

        >>> import toponetx as tnx
        >>> from topoembedx.classes.cell2vec import Cell2Vec
        >>> domain = tnx.CellComplex([[0, 1, 2]])
        >>> model = Cell2Vec(
        ...     dimensions=2,
        ...     walk_number=2,
        ...     walk_length=4,
        ...     window_size=2,
        ...     workers=1,
        ... )
        >>> model.fit(
        ...     domain,
        ...     neighborhood_type="adj",
        ...     neighborhood_dim={"rank": 0},
        ... )
        >>> model.get_embedding().shape
        (3, 2)

        Fit on an incidence-based connection graph:

        >>> model = Cell2Vec(
        ...     dimensions=2,
        ...     walk_number=2,
        ...     walk_length=4,
        ...     window_size=2,
        ...     workers=1,
        ... )
        >>> model.fit(
        ...     domain,
        ...     neighborhood_type="connection",
        ...     neighborhood_dim={"rank": 0, "to_rank": 1},
        ... )
        >>> model.get_embedding().shape
        (6, 2)

        Fit on an augmented Hasse graph:

        >>> model = Cell2Vec(
        ...     dimensions=2,
        ...     walk_number=2,
        ...     walk_length=4,
        ...     window_size=2,
        ...     workers=1,
        ... )
        >>> model.fit(
        ...     domain,
        ...     neighborhood_type="augmented_hasse",
        ...     neighborhood_dim={
        ...         "ranks": [0, 1, 2],
        ...         "neighborhoods": [
        ...             {"type": "adj", "rank": 0},
        ...             {"type": "coadj", "rank": 1},
        ...         ],
        ...     },
        ... )
        >>> model.get_embedding().shape
        (7, 2)
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
            The learned cell embeddings. If ``get_dict`` is ``True``, the
            result is a dictionary keyed by the cells stored in ``self.ind``.
            Otherwise, the result is a NumPy array with one row per embedded
            cell and one column per embedding dimension.

        Examples
        --------
        Return embeddings as an array:

        >>> import toponetx as tnx
        >>> from topoembedx.classes.cell2vec import Cell2Vec
        >>> domain = tnx.CellComplex([[0, 1, 2]])
        >>> model = Cell2Vec(
        ...     dimensions=2,
        ...     walk_number=2,
        ...     walk_length=4,
        ...     window_size=2,
        ...     workers=1,
        ... )
        >>> model.fit(
        ...     domain,
        ...     neighborhood_type="connection",
        ...     neighborhood_dim={"rank": 0, "to_rank": 1},
        ... )
        >>> model.get_embedding().shape
        (6, 2)

        Return embeddings as a dictionary indexed by cell labels:

        >>> embedding = model.get_embedding(get_dict=True)
        >>> len(embedding) == len(model.ind)
        True
        """
        emb = super().get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb, strict=True))
        return emb
