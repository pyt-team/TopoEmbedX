"""Test Cell2Vec class."""

import numpy as np
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
