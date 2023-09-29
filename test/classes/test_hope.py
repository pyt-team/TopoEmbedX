"""Test HOPE class."""

import numpy as np
import pytest
import toponetx as tnx

from topoembedx.classes.hope import HOPE


class TestHOPE:
    """Test HigherOrderLaplacianEigenmaps class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        cx = tnx.classes.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

        # Create a Cell2Vec object
        dc = HOPE(dimensions=2)

        # Fit the Cell2Vec object to the graph and get embedding for nodes (using adjacency matrix A0)
        dc.fit(cx, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1})

        # Check that the shape of the embedding is correct
        assert dc.get_embedding().shape == (len(cx.nodes), 2)

        # Check that the shape of the embedding dictionary is correct
        ind = dc.get_embedding(get_dict=True)
        assert (len(ind)) == len(cx.nodes)

        A = cx.adjacency_matrix(0)

        eigen_vec, eigen_val = HOPE._laplacian_pe(A, k=4, return_eigenval=True)

        assert len(eigen_val) != 0
