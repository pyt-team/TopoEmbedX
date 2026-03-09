"""Test TopoRep class."""

import numpy as np
import toponetx as tnx

from topoembedx.classes.toporep import TopoRep


class TestTopoRep:
    """Test TopoRep class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        # Create a small complex
        SC = tnx.SimplicialComplex([[0, 1], [1, 2, 3, 4], [3, 4, 5, 6, 7, 8]])

        # Create a TopoRep object
        toporep = TopoRep(dimensions=3, order=2)

        # Fit the TopoRep object to the graph and get embedding for nodes (using adjacency matrix A0)
        toporep.fit(
            SC, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": 1}
        )

        # Check that the shape of the embedding is correct
        assert toporep.get_embedding().shape == (len(SC.nodes), 3 * 2)

        # Check that the shape of the embedding dictionary is correct
        ind = toporep.get_embedding(get_dict=True)
        assert (len(ind)) == len(SC.nodes)

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(toporep.get_embedding()[0], toporep.get_embedding()[1])
