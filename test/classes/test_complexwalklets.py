"""Test ComplexWalklets class."""

import numpy as np
import toponetx as tnx

from topoembedx.classes.complexwalklets import ComplexWalklets


class TestComplexWalklets:
    """Test ComplexWalklets class."""

    def test_fit_and_get_embedding(self):
        """Test fitting on adjacency neighborhoods."""
        SC = tnx.SimplicialComplex([[0, 1], [1, 2, 3, 4], [3, 4, 5, 6, 7, 8]])

        complexwalklets = ComplexWalklets(
            dimensions=3,
            window_size=2,
            walk_number=4,
            walk_length=8,
            workers=1,
            epochs=1,
            seed=42,
        )

        complexwalklets.fit(
            SC, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": 1}
        )

        assert complexwalklets.get_embedding().shape == (len(SC.nodes), 3 * 2)

        ind = complexwalklets.get_embedding(get_dict=True)
        assert len(ind) == len(SC.nodes)

        assert not np.allclose(
            complexwalklets.get_embedding()[0], complexwalklets.get_embedding()[1]
        )

    def test_fit_and_get_embedding_on_coadjacency(self):
        """Test fitting on coadjacency neighborhoods."""
        SC = tnx.SimplicialComplex([[0, 1, 2], [1, 2, 3]])

        complexwalklets = ComplexWalklets(
            dimensions=2,
            window_size=2,
            walk_number=4,
            walk_length=8,
            workers=1,
            epochs=1,
            seed=7,
        )

        complexwalklets.fit(SC, neighborhood_type="coadj", neighborhood_dim={"rank": 1})

        embedding = complexwalklets.get_embedding()
        assert embedding.shape == (len(complexwalklets.ind), 2 * 2)

        ind = complexwalklets.get_embedding(get_dict=True)
        assert len(ind) == len(complexwalklets.ind)

        assert not np.allclose(embedding[0], embedding[1])
