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

        dc = Cell2Vec(dimensions=5)

        dc.fit(
            cx,
            neighborhood_type="adj",
            neighborhood_dim={"rank": 0, "via_rank": -1},
        )

        assert dc.get_embedding().shape == (len(cx.nodes), 5)

        ind = dc.get_embedding(get_dict=True)
        assert len(ind) == len(cx.nodes)

        assert not np.allclose(dc.get_embedding()[0], dc.get_embedding()[1])

    def test_fit_with_coadjacency_neighborhood(self):
        """Test Cell2Vec with a coadjacency neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="coadj",
            neighborhood_dim={"rank": 1, "via_rank": -1},
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (3, 2)
        assert len(embedding_dict) == 3
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_connection_neighborhood(self):
        """Test Cell2Vec with a connection neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (6, 2)
        assert len(embedding_dict) == 6
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_inc_neighborhood(self):
        """Test Cell2Vec with the inc neighborhood alias."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="inc",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (6, 2)
        assert len(embedding_dict) == 6
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_incidence_neighborhood(self):
        """Test Cell2Vec with the incidence neighborhood alias."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="incidence",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (6, 2)
        assert len(embedding_dict) == 6
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_hasse_neighborhood(self):
        """Test Cell2Vec with a Hasse neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (7, 2)
        assert len(embedding_dict) == 7
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_augmented_hasse_neighborhood(self):
        """Test Cell2Vec with an augmented Hasse neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
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

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (7, 2)
        assert len(embedding_dict) == 7
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_combinatorial_arbitrary_connection(self):
        """Test Cell2Vec with an arbitrary B_ij connection neighborhood."""
        cx = self._small_combinatorial_complex()

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 2},
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (4, 2)
        assert len(embedding_dict) == 4
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_combinatorial_augmented_hasse(self):
        """Test Cell2Vec with an augmented Hasse combinatorial complex graph."""
        cx = self._small_combinatorial_complex()

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
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

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (7, 2)
        assert len(embedding_dict) == 7
        assert set(embedding_dict) == set(model.ind)

    def test_fit_with_directed_connection_neighborhood(self):
        """Test Cell2Vec with a directed connection neighborhood."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={
                "rank": 0,
                "to_rank": 1,
                "symmetric": False,
            },
        )

        embedding = model.get_embedding()
        embedding_dict = model.get_embedding(get_dict=True)

        assert embedding.shape == (6, 2)
        assert len(embedding_dict) == 6
        assert set(embedding_dict) == set(model.ind)

    def test_graph_from_adjacency_is_unweighted(self):
        """Test graph construction removes edge weights for KarateClub."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )
        model.fit(
            cx,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        graph = Cell2Vec._graph_from_adjacency(model.A)

        assert graph.number_of_nodes() == model.A.shape[0]
        assert graph.number_of_edges() >= model.A.nnz // 2
        assert all(data == {} for _, _, data in graph.edges(data=True))

    def test_fit_with_invalid_neighborhood_type_raises_error(self):
        """Test Cell2Vec rejects invalid neighborhood types."""
        cx = tnx.CellComplex([[0, 1, 2]])

        model = Cell2Vec(
            dimensions=2,
            walk_number=2,
            walk_length=4,
            window_size=2,
            workers=1,
        )

        with pytest.raises(TypeError) as e:
            model.fit(cx, neighborhood_type="wrong")

        assert "Input neighborhood_type must be one of" in str(e.value)

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
