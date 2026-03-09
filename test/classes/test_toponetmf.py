"""Test TopoNetMF class."""

import numpy as np
import toponetx as tnx

from topoembedx.classes.toponetmf import TopoNetMF


class TestTopoNetMF:
    """Test TopoNetMF class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        # Create a small complex
        SC = tnx.SimplicialComplex([[0, 1], [1, 2, 3, 4], [3, 4, 5, 6, 7, 8]])

        # Create a TopoNetMF object
        toponetmf = TopoNetMF(dimensions=3)

        # Fit the TopoNetMF object to the graph and get embedding for nodes (using adjacency matrix A0)
        toponetmf.fit(
            SC, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": 1}
        )

        # Check that the shape of the embedding is correct
        assert toponetmf.get_embedding().shape == (len(SC.nodes), 3)

        # Check that the shape of the embedding dictionary is correct
        ind = toponetmf.get_embedding(get_dict=True)
        assert (len(ind)) == len(SC.nodes)

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(
            toponetmf.get_embedding()[0], toponetmf.get_embedding()[1]
        )
