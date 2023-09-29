"""Testing the neighborhood module."""

import pytest
import toponetx as tnx

import topoembedx as tex


class TestNeighborhood:
    """Test the neighborhood module of TopoEmbedX."""

    def test_neighborhood_from_complex_raise_error(self):
        """Testing if right assertion is raised for incorrect type."""
        with pytest.raises(TypeError) as e:
            tex.neighborhood.neighborhood_from_complex(1)

        assert (
            str(e.value)
            == """Input Complex can only be a SimplicialComplex, CellComplex, PathComplex ColoredHyperGraph or CombinatorialComplex."""
        )

    def test_neighborhood_from_complex_matrix_dimension_cell_complex(self):
        """Testing the matrix dimensions for the adjacency and coadjacency matrices."""
        # Testing for the case of Cell Complex
        cc1 = tnx.classes.CellComplex(
            [[0, 1, 2, 3], [1, 2, 3, 4], [1, 3, 4, 5, 6, 7, 8]]
        )

        cc2 = tnx.classes.CellComplex([[0, 1, 2], [1, 2, 3]])

        ind, A = tex.neighborhood.neighborhood_from_complex(cc1)
        assert A.todense().shape == tuple([9, 9])
        assert len(ind) == 9

        ind, A = tex.neighborhood.neighborhood_from_complex(cc2)
        assert A.todense().shape == tuple([4, 4])
        assert len(ind) == 4

        ind, A = tex.neighborhood.neighborhood_from_complex(
            cc1, neighborhood_type="coadj"
        )
        assert A.todense().shape == tuple([9, 9])
        assert len(ind) == 9

        ind, A = tex.neighborhood.neighborhood_from_complex(
            cc2, neighborhood_type="coadj"
        )
        assert A.todense().shape == tuple([4, 4])
        assert len(ind) == 4
