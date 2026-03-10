"""Test TopoRandNE class."""

import numpy as np
import toponetx as tnx

from topoembedx.classes.toporandne import TopoRandNE


class TestTopoRandNE:
    """Test TopoRandNE class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        # Create a small complex
        SC = tnx.SimplicialComplex([[0, 1], [1, 2, 3, 4], [3, 4, 5, 6, 7, 8]])

        # Create a TopoRandNE object
        toporandne = TopoRandNE(dimensions=3)

        # Fit the TopoRandNE object to the graph and get embedding for nodes (using adjacency matrix A0)
        toporandne.fit(
            SC, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": 1}
        )

        # Check that the shape of the embedding is correct
        assert toporandne.get_embedding().shape == (len(SC.nodes), 3)

        # Check that the shape of the embedding dictionary is correct
        ind = toporandne.get_embedding(get_dict=True)
        assert (len(ind)) == len(SC.nodes)

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(
            toporandne.get_embedding()[0], toporandne.get_embedding()[1]
        )
