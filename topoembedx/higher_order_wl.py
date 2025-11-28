"""Higher-order Weisfeiler–Lehman hashing on complexes."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable
from typing import Any, Literal

import networkx as nx
import toponetx as tnx
from scipy.sparse import csr_matrix


class HigherOrderWeisfeilerLehmanHashing:
    r"""
    Higher-order Weisfeiler–Lehman (WL) feature extractor on TopoNetX complexes.

    This class generalizes the classical Weisfeiler–Lehman hashing from graphs
    to arbitrary topological domains supported by TopoNetX
    (CellComplex, SimplicialComplex, CombinatorialComplex, PathComplex, ColoredHyperGraph).

    Instead of a graph adjacency, it uses an arbitrary (co)adjacency neighborhood
    matrix computed from the complex (e.g. adjacency of 0-cells, coadjacency of 1-cells,
    adjacency via another rank, etc.).

    Mathematically, we consider:
        - A complex :math:`\mathcal{K}`.
        - A neighborhood matrix :math:`A` over a chosen family of cells (e.g. rank-0, rank-1, ...).
        - WL is then run on the graph :math:`G = (V, E)` with
          :math:`V = \{0, \dots, n-1\}` indexing those cells, and
          edges induced by the non-zero pattern of :math:`A`.

    Parameters
    ----------
    wl_iterations : int
        Number of WL iterations (depth of refinement).
    erase_base_features : bool, default=False
        If True, drop the base features (iteration 0) from the final feature lists.

    Notes
    -----
    - The base features can be:
        * explicitly provided via ``cell_features`` in :meth:`fit`, or
        * default to the degree in the neighborhood graph induced by the
          chosen (co)adjacency matrix.
    - After fitting, you can obtain:
        * cell-level features via :meth:`get_cell_features`
        * complex-level (bag-of-features) representation via :meth:`get_domain_features`.
    """

    # Neighborhood matrix and index list
    A: csr_matrix
    ind: list[Hashable]

    def __init__(
        self, wl_iterations: int = 2, erase_base_features: bool = False
    ) -> None:
        self.wl_iterations = wl_iterations
        self.erase_base_features = erase_base_features

        # Will be populated by fit()
        self.domain: tnx.Complex | None = None
        self.neighborhood_type: str | None = None
        self.neighborhood_dim: dict[str, int] | None = None

        self.graph_: nx.Graph | None = None
        self._index_to_cell: dict[int, Hashable] = {}
        self._cell_to_index: dict[Hashable, int] = {}

        # WL internal state
        self.features: dict[int, Any] = {}
        self.extracted_features: dict[int, list[str]] = {}

    # -------------------------------------------------------------------------
    # Core public API
    # -------------------------------------------------------------------------
    def fit(
        self,
        domain: tnx.Complex,
        neighborhood_type: Literal["adj", "coadj"] = "adj",
        neighborhood_dim: dict[str, int] | None = None,
        cell_features: dict[Hashable, Any] | None = None,
    ) -> HigherOrderWeisfeilerLehmanHashing:
        r"""
        Fit the higher-order WL hashing on a complex.

        Parameters
        ----------
        domain : toponetx.classes.Complex
            A complex object. The complex can be one of:
            - CellComplex
            - CombinatorialComplex
            - PathComplex
            - SimplicialComplex
            - ColoredHyperGraph
        neighborhood_type : {"adj", "coadj"}, default="adj"
            The type of neighborhood to compute:
            - "adj"   → adjacency matrix
            - "coadj" → coadjacency matrix
        neighborhood_dim : dict, optional
            Integer parameters specifying the neighborhood of the cells to generate features.
            Follows the same convention as :func:`neighborhood_from_complex`:

            - For Cell/Simplicial/Path complexes, the (co)adjacency is specified by:
                  neighborhood_dim["rank"]
            - For Combinatorial/ColoredHyperGraph, it is specified by:
                  neighborhood_dim["rank"], neighborhood_dim["via_rank"]
        cell_features : dict, optional
            Optional base features for the cells. If provided, this must be a dictionary
            mapping each cell identifier in the chosen domain rank to a scalar or
            hashable attribute.

            The keys must match the entries of the index list ``ind`` returned by
            :func:`neighborhood_from_complex`. If None, the base features default to
            the degrees in the induced neighborhood graph.

        Returns
        -------
        HigherOrderWeisfeilerLehmanHashing
            The fitted instance (for chaining).
        """
        self.domain = domain
        self.neighborhood_type = neighborhood_type
        self.neighborhood_dim = neighborhood_dim

        # 1. Build neighborhood matrix and index list
        self.ind, self.A = neighborhood_from_complex(
            domain,
            neighborhood_type=neighborhood_type,
            neighborhood_dim=neighborhood_dim,
        )

        # 2. Build an internal graph on indices 0..n-1
        #    use the sparsity pattern of A as adjacency
        g = nx.from_scipy_sparse_array(self.A)
        # Optionally add self-loops, mirroring Cell2Vec behavior
        g.add_edges_from((idx, idx) for idx in range(g.number_of_nodes()))

        self.graph_ = g
        self._index_to_cell = dict(enumerate(self.ind))
        self._cell_to_index = {cell: idx for idx, cell in self._index_to_cell.items()}

        # 3. Set base features and run WL recursions
        self._set_features(cell_features=cell_features)
        self._do_recursions()

        return self

    def get_index_features(self) -> dict[int, list[str]]:
        """
        Get WL feature sequences indexed by the internal integer index.

        Returns
        -------
        dict
            Mapping from internal index (0..n-1) to list of WL features
            across iterations, i.e.:
                {idx: [f^0(idx), f^1(idx), ..., f^T(idx)]}
        """
        return self.extracted_features

    def get_cell_features(self) -> dict[Hashable, list[str]]:
        """
        Get WL feature sequences for each cell in the chosen rank.

        Returns
        -------
        dict
            Mapping from cell identifier to list of WL features across iterations.
            The cell identifiers coincide with the entries of ``self.ind`` returned
            by :func:`neighborhood_from_complex`.
        """
        return {
            self._index_to_cell[idx]: feats
            for idx, feats in self.extracted_features.items()
        }

    def get_domain_features(self) -> list[str]:
        """
        Get a bag (multiset) of all WL features across all cells and iterations.

        Returns
        -------
        list of str
            Concatenation of all WL features, i.e. a global representation for the complex.
        """
        return [
            feature
            for idx, features in self.extracted_features.items()
            for feature in features
        ]

    # -------------------------------------------------------------------------
    # Internal helpers: WL logic
    # -------------------------------------------------------------------------
    def _set_features(self, cell_features: dict[Hashable, Any] | None = None) -> None:
        """
        Initialize base features for the WL recursion.
        """
        assert self.graph_ is not None, (
            "Graph has not been constructed. Call fit() first."
        )

        if cell_features is not None:
            # Map cell_features (keyed by cells) to internal index-based features
            features: dict[int, Any] = {}
            missing_cells = []
            for idx, cell in self._index_to_cell.items():
                if cell not in cell_features:
                    missing_cells.append(cell)
                else:
                    features[idx] = cell_features[cell]

            if missing_cells:
                # Show a small subset of missing cells for debugging
                preview = missing_cells[:5]
                raise ValueError(
                    "Provided cell_features is missing values for some cells. "
                    f"Example missing cells: {preview}"
                )

            self.features = features
        else:
            # Default base features: degree in the neighborhood graph
            self.features = {
                idx: self.graph_.degree(idx)  # type: ignore[arg-type]
                for idx in self.graph_.nodes
            }

        # Initialize extracted_features with the base labels as strings
        self.extracted_features = {idx: [str(v)] for idx, v in self.features.items()}

    def _erase_base_features(self) -> None:
        """Erase the base features (iteration 0) from the feature lists."""
        for idx in list(self.extracted_features.keys()):
            if self.extracted_features[idx]:
                del self.extracted_features[idx][0]

    def _do_a_recursion(self) -> dict[int, str]:
        """
        Perform a single WL refinement step.

        For each index i:
            new_feat(i) = hash( feat(i), multiset{ feat(j) : j in N(i) } )
        """
        assert self.graph_ is not None, (
            "Graph has not been constructed. Call fit() first."
        )

        new_features: dict[int, str] = {}

        for idx in self.graph_.nodes:
            neighbors = self.graph_.neighbors(idx)
            neigh_vals = [self.features[nb] for nb in neighbors]

            # Concatenate current feature with sorted neighbor features
            parts = [str(self.features[idx])] + sorted(str(v) for v in neigh_vals)
            concat = "_".join(parts)

            hash_obj = hashlib.md5(concat.encode())
            hashing = hash_obj.hexdigest()
            new_features[idx] = hashing

        # Append this iteration's feature to the history
        self.extracted_features = {
            idx: self.extracted_features[idx] + [feat]
            for idx, feat in new_features.items()
        }

        return new_features

    def _do_recursions(self) -> None:
        """Run all WL iterations."""
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()

        if self.erase_base_features:
            self._erase_base_features()
