"""Weisfeiler-Lehman algorithms for higher-order topological domains."""

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any

import networkx as nx
import numpy as np
import toponetx as tnx

__all__ = ["SimplicialWeisfeilerLehman", "WeisfeilerLehman"]


def _check_convergence(
    old_colors: np.ndarray[Any, np.dtype[np.uint]],
    new_colors: np.ndarray[Any, np.dtype[np.uint]],
) -> bool:
    """Check if the coloring has converged.

    Coloring has converged if any pair of elements with the same color in
    ``old_colors`` also have the same color in ``new_colors`` (but not necessarily the
    same color as before).

    Parameters
    ----------
    old_colors : np.ndarray
        The previous coloring.
    new_colors : np.ndarray
        The current coloring.

    Returns
    -------
    bool
        True if the coloring has converged, False otherwise.
    """
    color_to_indices: dict[int, list[int]] = {}
    for idx, color in enumerate(old_colors):
        color_to_indices.setdefault(color, []).append(idx)

    for indices in color_to_indices.values():
        reference_color = new_colors[indices[0]]
        for idx in indices[1:]:
            if new_colors[idx] != reference_color:
                return False
    return True


class BaseWeisfeilerLehman(ABC):
    """Abstract base class for Weisfeiler-Lehman color refinement algorithms.

    This class provides common functionality for implementing the Weisfeiler-Lehman
    algorithm on different structures.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of refinement iterations to perform. If not given, colors are
        refined until convergence.

    Attributes
    ----------
    n_iter_ : int
        Number of iterations performed during the last fit.
    color_map_ : dict[tuple[Hashable, ...], int]
        Mapping from color signatures to integer color identifiers.
    """

    max_iter: int | None

    color_map_: dict[tuple[Hashable, ...], int]
    n_iter_: int
    _next_color: int

    def __init__(
        self,
        max_iter: int | None = None,
    ) -> None:
        self.max_iter = max_iter
        self._reset()

    def _reset(self) -> None:
        """Reset the internal state of the algorithm."""
        self.color_map_ = {}
        self.n_iter_ = 0
        self._next_color = 0

    def _get_color(self, signature: tuple[Hashable, ...]) -> int:
        """Get or create a color for a given signature.

        Parameters
        ----------
        signature : tuple[Hashable, ...]
            The signature to get a color for.

        Returns
        -------
        int
            The color identifier for the signature.
        """
        if signature not in self.color_map_:
            self.color_map_[signature] = self._next_color
            self._next_color += 1
        return self.color_map_[signature]

    @abstractmethod
    def refine(self, domain: Any, initial_coloring: np.ndarray | None = None):
        """Perform a single iteration of the refinement process.

        Parameters
        ----------
        domain : Any
            The domain to apply the Weisfeiler-Lehman algorithm on.
        initial_coloring : np.ndarray, optional
            Initial colors. If not provided, elements are initially colored uniformly.

        Returns
        -------
        np.ndarray
            Final coloring after one refinement step.
        """

    @abstractmethod
    def transform(self, X: Any):
        """Apply Weisfeiler-Lehman color refinement.

        Parameters
        ----------
        X : Any
            The domain to transform.

        Returns
        -------
        np.ndarray
            Final coloring after refinement iterations.
        """

    @abstractmethod
    def fit_transform(self, X: Any):
        """Fit the algorithm and transform the input in one step.

        Parameters
        ----------
        X : Any
            The domain to fit and transform.

        Returns
        -------
        np.ndarray
            Final coloring after refinement iterations.
        """


class SimplicialWeisfeilerLehman(BaseWeisfeilerLehman):
    """Simplicial version of the Weisfeiler-Lehman algorithm.

    This class implements the Weisfeiler-Lehman color refinement algorithm for
    simplicial complexes, providing a scikit-learn-like interface for feature
    extraction from topological structures.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of refinement iterations to perform. Refinement is stopped
        when either convergence is reached or the maximum number of iterations is hit.
        If not given, colors are always refined until convergence.

    Attributes
    ----------
    n_iter_ : int
        Number of iterations performed during the last fit.
    color_map_ : dict[tuple, int]
        Mapping from color signatures to integer color identifiers.

    Examples
    --------
    >>> import toponetx as tnx
    >>> import topoembedx as tex
    >>> sc = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 4]])
    >>> wl = tex.SimplicialWeisfeilerLehman(max_iter=3)
    >>> wl.fit_transform(sc)
    [1, 2, 3, 4]

    References
    ----------
    .. [1] Weisfeiler, B., Leman, A. (1968). The reduction of a graph to canonical form
       and the algebra which appears therein. nti, Series, 2(9), 12-16.
    .. [2] Bodnar, C., Frasca, F., Wang, Y., Otter, N., Montufar, G. F., Lio, P.,
       Bronstein, M. (2021, July). Weisfeiler and lehman go topological: Message
       passing simplicial networks. In International conference on machine learning
       (pp. 1026-1037). PMLR.
    """

    def __init__(self, max_iter: int | None = None) -> None:
        super().__init__(max_iter=max_iter)

    def _collect_neighbors(
        self, domain: tnx.SimplicialComplex, simplex: frozenset, rank: int
    ) -> dict[str, list[int]]:
        """Collect neighbor colors for a simplex.

        Parameters
        ----------
        domain : tnx.SimplicialComplex
            The simplicial complex.
        simplex : frozenset
            The simplex to collect neighbors for.
        rank : int
            The rank of the simplex.

        Returns
        -------
        dict[str, list[int]]
            Dictionary with keys 'boundary', 'coboundary', 'cofaces' containing
            lists of neighbor colors.
        """
        neighbors = {"boundary": [], "coboundary": [], "cofaces": []}

        # Boundary: (rank-1)-faces
        if rank > 0:
            for face in domain.get_boundaries(
                [simplex], min_dim=rank - 1, max_dim=rank - 1
            ):
                face_idx = self._simplex_to_idx[rank - 1][frozenset(face)]
                neighbors["boundary"].append(
                    int(self._current_colors[rank - 1][face_idx])
                )

        # Coboundary: (rank+1)-cofaces
        if rank < domain.dim - 1:
            for coface in domain.get_cofaces(simplex, rank + 1):
                coface_idx = self._simplex_to_idx[rank + 1][frozenset(coface)]
                neighbors["coboundary"].append(
                    int(self._current_colors[rank + 1][coface_idx])
                )

        return neighbors

    def _create_signature(
        self, current_color: int, neighbor_colors: dict[str, list[int]]
    ) -> tuple[Hashable, ...]:
        """Create a signature from current color and neighbor colors.

        Parameters
        ----------
        current_color : int
            The current color of the simplex.
        neighbor_colors : dict[str, list[int]]
            Dictionary of neighbor colors by type.

        Returns
        -------
        tuple
            The signature tuple.
        """
        signature_parts = [current_color]
        signature_parts.extend(
            tuple(sorted(neighbor_colors[key]))
            for key in ["boundary", "coboundary", "cofaces"]
        )
        return tuple(signature_parts)

    def refine(
        self,
        domain: tnx.SimplicialComplex,
        initial_coloring: list[np.ndarray[Any, np.dtype[np.uint]]] | None = None,
    ) -> list[np.ndarray[Any, np.dtype[np.uint]]]:
        """Perform a single iteration of the refinement process.

        Parameters
        ----------
        domain : tnx.SimplicialComplex
            The simplicial complex where to apply the Weisfeiler-Lehman algorithm on.
        initial_coloring : list[np.ndarray], optional
            Initial colors of all simplices in the domain. If not provided, simplices
            are initially colored in a single color.

        Returns
        -------
        list[np.ndarray]
            Final coloring of the simplices in the domain after one refinement step.

        Raises
        ------
        ValueError
            If the provided ``initial_coloring`` is not suitable for the ``domain``.
        """
        if initial_coloring is None:
            initial_coloring = [
                np.zeros(domain.shape[rank], dtype=int) for rank in range(domain.dim)
            ]

        if len(initial_coloring) != domain.dim:
            raise ValueError(
                "The length of the coloring list must match the dimension of the domain."
            )
        if any(len(c) != domain.shape[rank] for rank, c in enumerate(initial_coloring)):
            raise ValueError(
                "The length of the coloring list must match the shape of the domain."
            )

        self._current_colors = initial_coloring
        new_colors = []

        # Build index mappings for each rank
        self._simplex_to_idx = {}
        for rank in range(domain.dim):
            simplices = list(domain.skeleton(rank))
            self._simplex_to_idx[rank] = {
                frozenset(s): idx for idx, s in enumerate(simplices)
            }

        for rank in range(domain.dim):
            simplices = list(domain.skeleton(rank))
            rank_colors = np.zeros(len(simplices), dtype=int)

            for idx, simplex in enumerate(simplices):
                current_color = int(initial_coloring[rank][idx])
                neighbor_colors = self._collect_neighbors(
                    domain, frozenset(simplex), rank
                )
                signature = self._create_signature(current_color, neighbor_colors)
                rank_colors[idx] = self._get_color(signature)

            new_colors.append(rank_colors)

        return new_colors

    def transform(
        self, X: tnx.SimplicialComplex
    ) -> list[np.ndarray[Any, np.dtype[np.uint]]]:
        """Apply Weisfeiler-Lehman color refinement to a simplicial complex.

        Parameters
        ----------
        X : tnx.SimplicialComplex
            The simplicial complex to transform.

        Returns
        -------
        list[np.ndarray]
            Final coloring after refinement iterations.
            Each array corresponds to colors for simplices at a given rank.
        """
        colors = [np.zeros(X.shape[rank], dtype=np.uint) for rank in range(X.dim)]

        iteration = 0
        while True:
            new_colors = self.refine(X, colors)

            if all(np.array_equal(new_colors[r], colors[r]) for r in range(X.dim)):
                break

            colors = new_colors
            iteration += 1
            if self.max_iter is not None and iteration >= self.max_iter:
                break

        self.n_iter_ = iteration
        return colors

    def fit_transform(
        self, X: tnx.SimplicialComplex, y=None
    ) -> list[np.ndarray[Any, np.dtype[np.uint]]]:
        """Fit the algorithm and transform the input in one step.

        Parameters
        ----------
        X : tnx.SimplicialComplex
            The simplicial complex to fit and transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        list[np.ndarray]
            Final coloring after max_iter refinement iterations.
            Each array corresponds to colors for simplices at a given rank.
        """
        self._reset()
        return self.transform(X)


class WeisfeilerLehman(BaseWeisfeilerLehman):
    """Graph version of the Weisfeiler-Lehman algorithm.

    This class implements the Weisfeiler-Lehman color refinement algorithm for graphs,
    providing a scikit-learn-like interface for feature extraction from graph
    structures.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of refinement iterations to perform. Refinement is stopped
        when either convergence is reached or the maximum number of iterations is hit.
        If not given, colors are always refined until convergence.

    Attributes
    ----------
    n_iter_ : int
        Number of iterations performed during the last fit.
    color_map_ : dict[tuple, int]
        Mapping from color signatures to integer color identifiers.

    Examples
    --------
    >>> import networkx as nx
    >>> import topoembedx as tex
    >>> G = nx.karate_club_graph()
    >>> wl = tex.WeisfeilerLehman(max_iter=3)
    >>> colors = wl.fit_transform(G)

    References
    ----------
    .. [1] Weisfeiler, B., & Leman, A. (1968). "The reduction of a graph to
           canonical form and the algebra which appears therein."
    """

    def __init__(self, max_iter: int | None = None) -> None:
        super().__init__(max_iter=max_iter)

    def _create_signature(
        self, current_color: int, neighbor_colors: list[int]
    ) -> tuple[Hashable, ...]:
        """Create a signature from current color and neighbor colors.

        Parameters
        ----------
        current_color : int
            The current color of the node.
        neighbor_colors : list[int]
            List of neighbor colors.

        Returns
        -------
        tuple
            The signature tuple.
        """
        return (current_color, tuple(sorted(neighbor_colors)))

    def refine(
        self,
        domain: nx.Graph[Hashable],
        initial_coloring: np.ndarray[Any, np.dtype[np.uint]] | None = None,
    ) -> np.ndarray[Any, np.dtype[np.uint]]:
        """Perform a single iteration of the refinement process.

        Parameters
        ----------
        domain : networkx.Graph
            The graph to apply the Weisfeiler-Lehman algorithm on.
        initial_coloring : np.ndarray, optional
            Initial colors of all nodes in the graph. If not provided, nodes
            are initially colored in a single color.

        Returns
        -------
        np.ndarray
            Final coloring of the nodes after one refinement step.

        Raises
        ------
        ValueError
            If the provided ``initial_coloring`` is not suitable for the graph.
        """
        if initial_coloring is None:
            initial_coloring = np.zeros(domain.number_of_nodes(), dtype=np.uint)
        elif len(initial_coloring) != domain.number_of_nodes():
            raise ValueError(
                "The length of the coloring array must match the number of nodes."
            )

        node_to_idx = {node: idx for idx, node in enumerate(domain.nodes)}

        new_colors = np.zeros(domain.number_of_nodes(), dtype=int)
        for idx, node in enumerate(domain.nodes):
            neighbor_colors = [
                int(initial_coloring[node_to_idx[neighbor]])
                for neighbor in domain.neighbors(node)
            ]

            signature = self._create_signature(
                int(initial_coloring[idx]), neighbor_colors
            )
            new_colors[idx] = self._get_color(signature)

        return new_colors

    def transform(
        self, domain: nx.Graph[Hashable]
    ) -> np.ndarray[Any, np.dtype[np.uint]]:
        """Apply Weisfeiler-Lehman color refinement to a graph.

        Parameters
        ----------
        domain : networkx.Graph
            The graph to transform.

        Returns
        -------
        np.ndarray
            Final coloring after refinement iterations.
        """
        colors = np.zeros(len(domain.nodes()), dtype=np.uint)

        iteration = 0
        while True:
            new_colors = self.refine(domain, colors)

            if _check_convergence(colors, new_colors):
                break

            colors = new_colors
            iteration += 1
            if self.max_iter is not None and iteration >= self.max_iter:
                break

        self.n_iter_ = iteration
        return colors

    def fit_transform(self, X: nx.Graph[Hashable]) -> np.ndarray:
        """Fit the algorithm and transform the input in one step.

        Parameters
        ----------
        X : networkx.Graph
            The graph to fit and transform.

        Returns
        -------
        np.ndarray
            Final coloring after refinement iterations.
        """
        self._reset()
        return self.transform(X)
