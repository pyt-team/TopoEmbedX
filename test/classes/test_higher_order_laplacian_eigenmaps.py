"""Test HigherOrderLaplacianEigenmaps class."""

import numpy as np
import toponetx as tnx

from topoembedx.classes.higher_order_laplacian_eigenmaps import (
    HigherOrderLaplacianEigenmaps,
)


class TestHigherOrderLaplacianEigenmaps:
    """Test HigherOrderLaplacianEigenmaps class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        cx = tnx.classes.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

        # Create a Cell2Vec object
        dc = HigherOrderLaplacianEigenmaps(dimensions=5)

        # Fit the Cell2Vec object to the graph and get embedding for nodes (using adjacency matrix A0)
        dc.fit(
            cx, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": -1}
        )

        # Check that the shape of the embedding is correct
        assert dc.get_embedding().shape == (len(cx.nodes), 5)

        # Check that the shape of the embedding dictionary is correct
        ind = dc.get_embedding(get_dict=True)
        assert (len(ind)) == len(cx.nodes)

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(dc.get_embedding()[0], dc.get_embedding()[1])
