"""Test ComplexHeat class."""

import numpy as np
import pytest
import toponetx as tnx
from scipy.sparse import csr_matrix

from topoembedx.classes.complexheat import ComplexHeat


class TestComplexHeat:
    """Test ComplexHeat class."""

    def test_fit_and_get_embedding(self):
        """Test fitting on adjacency neighborhoods."""
        SC = tnx.SimplicialComplex([[0, 1], [1, 2, 3, 4], [3, 4, 5, 6, 7, 8]])

        complexheat = ComplexHeat(
            dimensions=4,
            diffusion_times=[0.1, 1.0],
            approximation_rank=4,
            seed=42,
        )
        complexheat.fit(
            SC, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": 1}
        )

        embedding = complexheat.get_embedding()
        assert embedding.shape == (len(SC.nodes), 4)

        ind = complexheat.get_embedding(get_dict=True)
        assert len(ind) == len(SC.nodes)

        assert not np.allclose(embedding[0], embedding[1])

    def test_fit_and_get_embedding_on_coadjacency(self):
        """Test fitting on coadjacency neighborhoods."""
        SC = tnx.SimplicialComplex([[0, 1, 2], [1, 2, 3]])

        complexheat = ComplexHeat(
            dimensions=5,
            diffusion_times=[0.5, 2.0],
            approximation_rank=2,
            seed=7,
        )
        complexheat.fit(SC, neighborhood_type="coadj", neighborhood_dim={"rank": 1})

        embedding = complexheat.get_embedding()
        assert embedding.shape == (len(complexheat.ind), 5)

        ind = complexheat.get_embedding(get_dict=True)
        assert len(ind) == len(complexheat.ind)

        assert not np.allclose(embedding[0], embedding[1])

    def test_padding_path(self):
        """Test zero-padding when raw features are narrower than dimensions."""
        SC = tnx.SimplicialComplex([[0, 1, 2]])

        complexheat = ComplexHeat(
            dimensions=6,
            diffusion_times=[1.0],
            approximation_rank=2,
            seed=1,
        )
        complexheat.fit(
            SC, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": 1}
        )

        embedding = complexheat.get_embedding()
        assert embedding.shape == (len(SC.nodes), 6)
        assert np.allclose(embedding[:, 2:], 0.0)

    def test_reduction_path(self):
        """Test dimensionality reduction when raw features are wider than dimensions."""
        SC = tnx.SimplicialComplex([[0, 1], [1, 2, 3], [3, 4, 5]])

        complexheat = ComplexHeat(
            dimensions=3,
            diffusion_times=[0.1, 1.0, 10.0],
            approximation_rank=5,
            seed=9,
        )
        complexheat.fit(
            SC, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": 1}
        )

        assert complexheat.get_embedding().shape == (len(SC.nodes), 3)

    def test_sparse_normalized_laplacian(self):
        """Test that the normalized Laplacian stays sparse."""
        SC = tnx.SimplicialComplex([[0, 1], [1, 2, 3]])
        complexheat = ComplexHeat(dimensions=2, approximation_rank=2, seed=0)
        _, adjacency = SC.adjacency_matrix(0, index=True)

        laplacian = complexheat._normalized_laplacian(adjacency)

        assert isinstance(laplacian, csr_matrix)

    def test_isolated_nodes_get_zero_diagonal_in_normalized_laplacian(self):
        """Test that isolated nodes have zero diagonal in the normalized Laplacian."""
        complexheat = ComplexHeat(dimensions=2, approximation_rank=2, seed=0)
        adjacency = csr_matrix(np.array([[0.0, 0.0], [0.0, 1.0]]))

        laplacian = complexheat._normalized_laplacian(adjacency).toarray()

        assert laplacian[0, 0] == 0.0

    def test_default_approximation_rank_resolution(self):
        """Test the default approximation rank resolution path."""
        complexheat = ComplexHeat(dimensions=4, approximation_rank=None, seed=0)

        assert complexheat._resolved_approximation_rank(100) == 32
        assert complexheat._resolved_approximation_rank(5) == 4

    def test_single_node_heat_signatures(self):
        """Test the single-node heat signature shortcut."""
        complexheat = ComplexHeat(
            dimensions=2,
            diffusion_times=[0.5, 2.0],
            approximation_rank=None,
            seed=0,
        )
        laplacian = csr_matrix([[0.0]])

        signatures = complexheat._approximate_heat_signatures(laplacian)

        expected = np.ones((1, 2))
        assert np.allclose(signatures, expected)

    def test_exact_width_resize_path(self):
        """Test returning features unchanged when widths already match."""
        complexheat = ComplexHeat(dimensions=3, approximation_rank=2, seed=0)
        features = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])

        resized = complexheat._resize_embedding(features)

        assert resized is features

    def test_raises_for_non_positive_dimensions(self):
        """Test that dimensions must be a positive integer."""
        with pytest.raises(ValueError, match="dimensions must be a positive integer"):
            ComplexHeat(dimensions=0)

    def test_raises_for_empty_diffusion_times(self):
        """Test that diffusion_times cannot be empty."""
        with pytest.raises(
            ValueError,
            match="`diffusion_times` must contain at least one positive value",
        ):
            ComplexHeat(diffusion_times=[])

    def test_raises_for_non_positive_diffusion_times(self):
        """Test that diffusion_times must contain only positive values."""
        with pytest.raises(
            ValueError,
            match="`diffusion_times` must contain only positive values",
        ):
            ComplexHeat(diffusion_times=[0.1, 0.0, 1.0])

        with pytest.raises(
            ValueError,
            match="`diffusion_times` must contain only positive values",
        ):
            ComplexHeat(diffusion_times=[0.1, -1.0])

    def test_raises_for_non_positive_approximation_rank(self):
        """Test that approximation_rank must be a positive integer."""
        with pytest.raises(
            ValueError, match="approximation_rank must be a positive integer"
        ):
            ComplexHeat(approximation_rank=0)
