"""Testing the neighborhood module."""

import pytest
import toponetx as tnx
from scipy.sparse import csr_matrix

from topoembedx.neighborhood import neighborhood_from_complex


class TestNeighborhood:
    """Test the neighborhood module of TopoEmbedX."""

    def test_neighborhood_from_complex_raise_error(self):
        """Testing if right assertion is raised for incorrect type."""
        with pytest.raises(TypeError) as e:
            neighborhood_from_complex(1)

        assert "Input Complex can only be" in str(e.value)

    def test_neighborhood_from_complex_invalid_neighborhood_type(self):
        """Testing if right assertion is raised for incorrect neighborhood type."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(TypeError) as e:
            neighborhood_from_complex(domain, neighborhood_type="wrong")

        assert "Input neighborhood_type must be one of" in str(e.value)

    def test_neighborhood_from_complex_matrix_dimension_cell_complex(self):
        """Testing matrix dimensions for adjacency and coadjacency matrices."""
        # Testing for the case of Cell Complex
        cc1 = tnx.classes.CellComplex(
            [[0, 1, 2, 3], [1, 2, 3, 4], [1, 3, 4, 5, 6, 7, 8]]
        )

        cc2 = tnx.classes.CellComplex([[0, 1, 2], [1, 2, 3]])

        ind, A = neighborhood_from_complex(cc1)
        assert A.todense().shape == (9, 9)
        assert len(ind) == 9

        ind, A = neighborhood_from_complex(cc2)
        assert A.todense().shape == (4, 4)
        assert len(ind) == 4

        ind, A = neighborhood_from_complex(cc1, neighborhood_type="coadj")
        assert A.todense().shape == (9, 9)
        assert len(ind) == 9

        ind, A = neighborhood_from_complex(cc2, neighborhood_type="coadj")
        assert A.todense().shape == (4, 4)
        assert len(ind) == 4

    def test_neighborhood_from_complex_connection_cell_complex(self):
        """Testing connection graph induced by a single incidence matrix."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind, A = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        assert isinstance(A, csr_matrix)
        assert A.shape == (len(ind), len(ind))
        assert A.shape == (6, 6)
        assert A.nnz > 0
        assert (A != A.T).nnz == 0

    def test_neighborhood_from_complex_incidence_aliases(self):
        """Testing incidence aliases for connection neighborhoods."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_inc, A_inc = neighborhood_from_complex(
            domain,
            neighborhood_type="inc",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )
        ind_incidence, A_incidence = neighborhood_from_complex(
            domain,
            neighborhood_type="incidence",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        assert ind_inc == ind_incidence
        assert A_inc.shape == A_incidence.shape
        assert (A_inc != A_incidence).nnz == 0

    def test_neighborhood_from_complex_hasse_cell_complex(self):
        """Testing Hasse graph construction over consecutive ranks."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind, A = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        assert isinstance(A, csr_matrix)
        assert A.shape == (len(ind), len(ind))
        assert A.shape == (7, 7)
        assert A.nnz > 0
        assert (A != A.T).nnz == 0

    def test_neighborhood_from_complex_multiple_rank_pairs(self):
        """Testing connection graph from multiple rank-pair matrices."""
        domain = self._small_combinatorial_complex()

        ind, A = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank_pairs": [(0, 1), (1, 2), (0, 2)]},
        )

        assert isinstance(A, csr_matrix)
        assert A.shape == (len(ind), len(ind))
        assert len(ind) == 7
        assert A.nnz > 0
        assert (A != A.T).nnz == 0

    def test_neighborhood_from_complex_arbitrary_bij_connection_graph(self):
        """Testing arbitrary B_ij connection graph for combinatorial complexes."""
        domain = self._small_combinatorial_complex()

        ind, A = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 2},
        )

        assert isinstance(A, csr_matrix)
        assert A.shape == (len(ind), len(ind))
        assert len(ind) == 4
        assert A.nnz > 0
        assert (A != A.T).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_cell_complex(self):
        """Testing augmented Hasse graph with extra same-rank neighborhoods."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_hasse, A_hasse = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )
        ind_augmented, A_augmented = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "adj", "rank": 0},
                    {"type": "coadj", "rank": 1},
                ],
            },
        )

        assert isinstance(A_augmented, csr_matrix)
        assert A_augmented.shape == (len(ind_augmented), len(ind_augmented))
        assert len(ind_augmented) == len(ind_hasse)
        assert A_augmented.shape == A_hasse.shape
        assert A_augmented.nnz >= A_hasse.nnz
        assert (A_augmented != A_augmented.T).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_combinatorial_complex(self):
        """Testing augmented Hasse graph with arbitrary cross-rank connections."""
        domain = self._small_combinatorial_complex()

        ind, A = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "rank_pairs": [(0, 1), (1, 2)],
                "neighborhoods": [
                    {"type": "connection", "rank": 0, "to_rank": 2},
                    {"type": "adj", "rank": 0, "via_rank": 1},
                    {"type": "coadj", "rank": 1, "via_rank": 0},
                ],
            },
        )

        assert isinstance(A, csr_matrix)
        assert A.shape == (len(ind), len(ind))
        assert len(ind) == 7
        assert A.nnz > 0
        assert (A != A.T).nnz == 0

    @staticmethod
    def _small_combinatorial_complex():
        """Create a small combinatorial complex with ranks 0, 1, and 2."""
        domain = tnx.classes.CombinatorialComplex()
        domain.add_cell([0], rank=0)
        domain.add_cell([1], rank=0)
        domain.add_cell([2], rank=0)
        domain.add_cell([0, 1], rank=1)
        domain.add_cell([1, 2], rank=1)
        domain.add_cell([0, 2], rank=1)
        domain.add_cell([0, 1, 2], rank=2)
        return domain
