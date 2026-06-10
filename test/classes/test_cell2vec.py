"""Test Cell2Vec class."""

import numpy as np
import pytest
import toponetx as tnx

from topoembedx.classes.cell2vec import Cell2Vec


class TestCell2Vec:
    """Test Cell2Vec class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        cx = tnx.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

        model = Cell2Vec(dimensions=5)

        model.fit(
            cx,
            neighborhood_type="adj",
            neighborhood_dim={"rank": 0, "via_rank": -1},
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (len(cx.nodes), 5)
        assert len(embedding_dict) == len(cx.nodes)
        assert set(embedding_dict) == set(model.ind)
        assert not np.allclose(embedding[0], embedding[1])

    def test_fit_with_coadjacency_neighborhood(self):
        """Test Cell2Vec with a coadjacency neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="coadj",
            neighborhood_dim={"rank": 1, "via_rank": -1},
        )

        self._assert_embedding(model, expected_rows=3, expected_columns=2)

    def test_fit_with_connection_neighborhood(self):
        """Test Cell2Vec with a connection neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        self._assert_embedding(model, expected_rows=6, expected_columns=2)

    def test_fit_with_inc_neighborhood(self):
        """Test Cell2Vec with the inc neighborhood alias."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="inc",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        self._assert_embedding(model, expected_rows=6, expected_columns=2)

    def test_fit_with_incidence_neighborhood(self):
        """Test Cell2Vec with the incidence neighborhood alias."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="incidence",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        self._assert_embedding(model, expected_rows=6, expected_columns=2)

    def test_fit_with_hasse_neighborhood(self):
        """Test Cell2Vec with a Hasse neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_augmented_hasse_neighborhood(self):
        """Test Cell2Vec with an augmented Hasse neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "adj", "rank": 0},
                    {"type": "coadj", "rank": 1},
                ],
            },
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_combinatorial_arbitrary_connection(self):
        """Test Cell2Vec with an arbitrary B_ij connection neighborhood."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 2},
        )

        self._assert_embedding(model, expected_rows=4, expected_columns=2)

    def test_fit_with_combinatorial_augmented_hasse(self):
        """Test Cell2Vec with an augmented Hasse combinatorial complex graph."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "rank_pairs": [(0, 1), (1, 2)],
                "neighborhoods": [
                    {"type": "connection", "rank": 0, "to_rank": 2},
                    {"type": "adj", "rank": 0, "via_rank": 1},
                    {"type": "coadj", "rank": 1, "via_rank": 0},
                ],
            },
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_directed_connection_neighborhood(self):
        """Test Cell2Vec with a directed connection neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={
                "rank": 0,
                "to_rank": 1,
                "symmetric": False,
            },
        )

        self._assert_embedding(model, expected_rows=6, expected_columns=2)

    def test_fit_with_directed_augmented_hasse_neighborhood(self):
        """Test Cell2Vec with a directed augmented Hasse neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "symmetric": False,
            },
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_ranked_labels_false_connection(self):
        """Test Cell2Vec with unranked labels in a connection graph."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={
                "rank": 0,
                "to_rank": 2,
                "ranked_labels": False,
            },
        )

        self._assert_embedding(model, expected_rows=4, expected_columns=2)
        assert all(not isinstance(cell, tuple) for cell in model.ind)

    def test_fit_with_multiple_rank_pairs_connection(self):
        """Test Cell2Vec with multiple rank-pair connection graphs."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank_pairs": [(0, 1), (1, 2), (0, 2)]},
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_pairs_alias_connection(self):
        """Test Cell2Vec with the pairs alias for rank pairs."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"pairs": [(0, 1), (1, 2), (0, 2)]},
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_ranks_connection(self):
        """Test Cell2Vec with ranks converted to connection rank pairs."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_target_rank_alias_connection(self):
        """Test Cell2Vec with target_rank as a connection alias."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "target_rank": 2},
        )

        self._assert_embedding(model, expected_rows=4, expected_columns=2)

    def test_fit_with_via_rank_alias_connection(self):
        """Test Cell2Vec with via_rank as a connection alias."""
        cx = self._small_combinatorial_complex()

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "via_rank": 2},
        )

        self._assert_embedding(model, expected_rows=4, expected_columns=2)

    def test_fit_with_augmented_hasse_inc_alias(self):
        """Test Cell2Vec with an inc entry inside augmented Hasse."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "inc", "rank": 0, "to_rank": 1},
                ],
            },
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_augmented_hasse_incidence_alias(self):
        """Test Cell2Vec with an incidence entry inside augmented Hasse."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "incidence", "rank": 1, "to_rank": 2},
                ],
            },
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_fit_with_augmented_hasse_hasse_entry(self):
        """Test Cell2Vec with a Hasse entry inside augmented Hasse."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "hasse", "ranks": [0, 1]},
                ],
            },
        )

        self._assert_embedding(model, expected_rows=7, expected_columns=2)

    def test_graph_from_adjacency_is_unweighted(self):
        """Test graph construction removes edge weights for KarateClub."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        graph = Cell2Vec._graph_from_adjacency(model.A)

        assert graph.number_of_nodes() == model.A.shape[0]
        assert graph.number_of_edges() >= model.A.nnz // 2
        assert all(data == {} for _, _, data in graph.edges(data=True))

    def test_graph_from_adjacency_adds_self_loops(self):
        """Test graph construction adds self-loops to all nodes."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        graph = Cell2Vec._graph_from_adjacency(model.A)

        assert all(graph.has_edge(index, index) for index in graph.nodes)

    def test_fit_with_invalid_neighborhood_type_raises_error(self):
        """Test Cell2Vec rejects invalid neighborhood types."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()

        with pytest.raises(TypeError) as e:
            model.fit(cx, neighborhood_type="wrong")

        assert "Input neighborhood_type must be one of" in str(e.value)

    def test_fit_with_invalid_neighborhood_dim_raises_error(self):
        """Test Cell2Vec propagates invalid neighborhood dimensions."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = self._small_model()

        with pytest.raises(ValueError) as e:
            model.fit(
                cx,
                neighborhood_type="connection",
                neighborhood_dim={"rank_pairs": []},
            )

        assert "At least one rank pair must be specified." in str(e.value)

    @staticmethod
    def _small_model():
        """Create a small Cell2Vec model for tests.

        Returns
        -------
        topoembedx.classes.cell2vec.Cell2Vec
            A Cell2Vec model with small random-walk and embedding settings.
        """
        return Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )

    @staticmethod
    def _assert_embedding(model, expected_rows, expected_columns):
        """Assert that a fitted Cell2Vec model has the expected embedding.

        Parameters
        ----------
        model : topoembedx.classes.cell2vec.Cell2Vec
            Fitted Cell2Vec model.
        expected_rows : int
            Expected number of embedded cells.
        expected_columns : int
            Expected embedding dimension.
        """
        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (expected_rows, expected_columns)
        assert len(embedding_dict) == expected_rows
        assert set(embedding_dict) == set(model.ind)

    @staticmethod
    def _small_combinatorial_complex():
        """Create a small combinatorial complex with ranks 0, 1, and 2.

        Returns
        -------
        toponetx.classes.CombinatorialComplex
            A small combinatorial complex.
        """
        domain = tnx.classes.CombinatorialComplex()
        domain.add_cell([0], rank=0)
        domain.add_cell([1], rank=0)
        domain.add_cell([2], rank=0)
        domain.add_cell([0, 1], rank=1)
        domain.add_cell([1, 2], rank=1)
        domain.add_cell([0, 2], rank=1)
        domain.add_cell([0, 1, 2], rank=2)
        return domain
