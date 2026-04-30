"""Test ComplexHeat class."""

import numpy as np
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
