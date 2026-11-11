"""Tests for Weisfeiler-Lehman algorithms."""

import networkx as nx
import numpy as np
import pytest
import toponetx as tnx

from topoembedx.classes.wl import SimplicialWeisfeilerLehman, WeisfeilerLehman


class TestWeisfeilerLehman:
    """Tests for the WeisfeilerLehman class."""

    def test_initialization(self) -> None:
        """Test WeisfeilerLehman initialization."""
        wl = WeisfeilerLehman()
        assert wl.max_iter is None

        wl = WeisfeilerLehman(max_iter=5)
        assert wl.max_iter == 5

    def test_complete_graph(self) -> None:
        """Test fit_transform on a complete graph."""
        G = nx.complete_graph(5)
        wl = WeisfeilerLehman()
        colors = wl.fit_transform(G)

        # Complete graph should converge immediately (all nodes are equivalent)
        assert np.all(colors == colors[0])
        assert wl.n_iter_ == 0

    def test_path_graph(self) -> None:
        """Test fit_transform on a path graph."""
        G = nx.path_graph(4)
        wl = WeisfeilerLehman()
        colors = wl.fit_transform(G)

        assert wl.n_iter_ == 1
        assert isinstance(colors, np.ndarray)
        assert colors.dtype == np.int_
        np.testing.assert_allclose(colors, np.array([0, 1, 1, 0]))

    def test_star_graph(self) -> None:
        """Test fit_transform on a star graph."""
        G = nx.star_graph(5)
        wl = WeisfeilerLehman(max_iter=3)
        colors = wl.fit_transform(G)

        assert wl.n_iter_ == 1
        assert isinstance(colors, np.ndarray)
        assert colors.dtype == np.int_
        np.testing.assert_allclose(colors, np.array([0, 1, 1, 1, 1]))

    def test_karate_club(self) -> None:
        """Test fit_transform on karate club graph."""
        G = nx.karate_club_graph()
        wl = WeisfeilerLehman(max_iter=3)
        colors = wl.fit_transform(G)

        assert len(colors) == 34
        assert wl.n_iter_ <= 3

    def test_fit_then_transform(self) -> None:
        """Test separate fit and transform calls."""
        G = nx.path_graph(4)
        wl = WeisfeilerLehman(max_iter=2)
        wl.fit(G)
        colors = wl.transform(G)

        assert len(colors) == 4

    def test_refine_with_initial_coloring(self) -> None:
        """Test refine with custom initial coloring."""
        G = nx.path_graph(4)
        wl = WeisfeilerLehman()
        initial = np.array([0, 1, 1, 0])
        colors = wl.refine(G, initial)

        assert len(colors) == 4

    def test_refine_invalid_coloring_length(self) -> None:
        """Test refine with invalid coloring length."""
        G = nx.path_graph(4)
        wl = WeisfeilerLehman()
        initial = np.array([0, 1])  # Wrong length

        with pytest.raises(ValueError, match="length of the coloring array"):
            wl.refine(G, initial)

    def test_reset(self) -> None:
        """Test that reset clears internal state."""
        G = nx.path_graph(4)
        wl = WeisfeilerLehman(max_iter=2)
        wl.fit_transform(G)

        assert len(wl.color_map_) > 0
        wl._reset()

        assert len(wl.color_map_) == 0
        assert wl.n_iter_ == 0
        assert wl._next_color == 0


class TestSimplicialWeisfeilerLehman:
    """Tests for the SimplicialWeisfeilerLehman class."""

    def test_initialization(self) -> None:
        """Test SimplicialWeisfeilerLehman initialization."""
        wl = SimplicialWeisfeilerLehman()
        assert wl.max_iter is None

        wl = SimplicialWeisfeilerLehman(max_iter=5)
        assert wl.max_iter == 5

    def test_fit_transform_simple_complex(self) -> None:
        """Test fit_transform on a simple simplicial complex."""
        sc = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 4]])
        wl = SimplicialWeisfeilerLehman(max_iter=3)
        colors = wl.fit_transform(sc)

        assert isinstance(colors, list)
        assert len(colors) == sc.dim
        for rank, rank_colors in enumerate(colors):
            assert len(rank_colors) == sc.shape[rank]

    def test_fit_transform_triangle(self) -> None:
        """Test fit_transform on a triangle."""
        sc = tnx.SimplicialComplex([[1, 2, 3]])
        wl = SimplicialWeisfeilerLehman(max_iter=2)
        colors = wl.fit_transform(sc)

        assert len(colors) == 2  # 0-simplices and 1-simplices
        assert wl.n_iter_ <= 2

    def test_convergence(self) -> None:
        """Test that algorithm converges."""
        sc = tnx.SimplicialComplex([[1, 2], [2, 3], [3, 1]])
        wl = SimplicialWeisfeilerLehman()
        colors = wl.fit_transform(sc)

        assert len(colors) == 2

    def test_fit_then_transform(self) -> None:
        """Test separate fit and transform calls."""
        sc = tnx.SimplicialComplex([[1, 2, 3]])
        wl = SimplicialWeisfeilerLehman(max_iter=2)
        wl.fit(sc)
        colors = wl.transform(sc)

        assert len(colors) == 2

    def test_refine_with_initial_coloring(self) -> None:
        """Test refine with custom initial coloring."""
        sc = tnx.SimplicialComplex([[1, 2, 3]])
        wl = SimplicialWeisfeilerLehman()
        initial = [
            np.array([0, 1, 2]),  # 0-simplices
            np.array([0, 0, 1]),  # 1-simplices
        ]
        colors = wl.refine(sc, initial)

        assert len(colors) == 2

    def test_refine_invalid_coloring_length(self) -> None:
        """Test refine with invalid coloring dimension."""
        sc = tnx.SimplicialComplex([[1, 2, 3]])
        wl = SimplicialWeisfeilerLehman()
        initial = [np.array([0, 1, 2])]  # Wrong dimension

        with pytest.raises(ValueError, match="length of the coloring list"):
            wl.refine(sc, initial)

    def test_refine_invalid_coloring_shape(self) -> None:
        """Test refine with invalid coloring shape."""
        sc = tnx.SimplicialComplex([[1, 2, 3]])
        wl = SimplicialWeisfeilerLehman()
        initial = [
            np.array([0, 1]),  # Wrong shape
            np.array([0, 0, 1]),
        ]

        with pytest.raises(ValueError, match="length of the coloring list"):
            wl.refine(sc, initial)

    def test_reset(self) -> None:
        """Test that reset clears internal state."""
        sc = tnx.SimplicialComplex([[1, 2, 3]])
        wl = SimplicialWeisfeilerLehman(max_iter=2)
        wl.fit_transform(sc)

        assert len(wl.color_map_) > 0
        wl._reset()

        assert len(wl.color_map_) == 0
        assert wl.n_iter_ == 0
        assert wl._next_color == 0

    def test_collect_neighbors(self) -> None:
        """Test neighbor collection for simplices."""
        sc = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 4]])
        wl = SimplicialWeisfeilerLehman()

        # Initialize for testing
        colors = [np.zeros(sc.shape[rank], dtype=int) for rank in range(sc.dim)]
        wl._current_colors = colors
        wl._simplex_to_idx = {}
        for rank in range(sc.dim):
            simplices = list(sc.skeleton(rank))
            wl._simplex_to_idx[rank] = {
                frozenset(s): idx for idx, s in enumerate(simplices)
            }

        # Test collecting neighbors for a 1-simplex
        simplex = frozenset([1, 2])
        neighbors = wl._collect_neighbors(sc, simplex, 1)

        assert "boundary" in neighbors
        assert "coboundary" in neighbors
        assert "cofaces" in neighbors
