"""Test HOPE class."""

import toponetx as tnx

from topoembedx.classes.hope import HOPE


class TestHOPE:
    """Test HOPE class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        cx = tnx.classes.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

        # Create a HOPE object
        dc = HOPE(dimensions=20)

        # Fit the HOPE object to the graph and get embedding for nodes (using adjacency matrix A0)
        dc.fit(cx, neighborhood_type="adj", neighborhood_dim={"rank": 0, "to_rank": -1})

        # Check that the shape of the embedding is correct
        assert dc.get_embedding().shape == (len(cx.nodes), 20)

        # Check that the shape of the embedding dictionary is correct
        ind = dc.get_embedding(get_dict=True)
        assert (len(ind)) == len(cx.nodes)

        A = cx.adjacency_matrix(0)

        eigvec, eigval = HOPE._laplacian_pe(A, n_eigvecs=4, return_eigenval=True)

        assert len(eigval) != 0
