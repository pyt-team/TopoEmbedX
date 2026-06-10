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

        assert (
            str(e.value)
            == "Input Complex can only be a SimplicialComplex, CellComplex, "
            "PathComplex ColoredHyperGraph or CombinatorialComplex."
        )

    def test_neighborhood_from_complex_invalid_neighborhood_type(self):
        """Testing if right assertion is raised for incorrect neighborhood type."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(TypeError) as e:
            neighborhood_from_complex(domain, neighborhood_type="wrong")

        assert "Input neighborhood_type must be one of" in str(e.value)

    def test_neighborhood_from_complex_matrix_dimension_cell_complex(self):
        """Testing matrix dimensions for adjacency and coadjacency matrices."""
        cc1 = tnx.classes.CellComplex(
            [[0, 1, 2, 3], [1, 2, 3, 4], [1, 3, 4, 5, 6, 7, 8]]
        )

        cc2 = tnx.classes.CellComplex([[0, 1, 2], [1, 2, 3]])

        ind, matrix = neighborhood_from_complex(cc1)
        assert matrix.todense().shape == (9, 9)
        assert len(ind) == 9

        ind, matrix = neighborhood_from_complex(cc2)
        assert matrix.todense().shape == (4, 4)
        assert len(ind) == 4

        ind, matrix = neighborhood_from_complex(cc1, neighborhood_type="coadj")
        assert matrix.todense().shape == (9, 9)
        assert len(ind) == 9

        ind, matrix = neighborhood_from_complex(cc2, neighborhood_type="coadj")
        assert matrix.todense().shape == (4, 4)
        assert len(ind) == 4

    def test_neighborhood_from_complex_combinatorial_same_rank(self):
        """Testing same-rank neighborhoods for combinatorial complexes."""
        domain = self._small_combinatorial_complex()

        ind_adj, matrix_adj = neighborhood_from_complex(
            domain,
            neighborhood_type="adj",
            neighborhood_dim={"rank": 0, "via_rank": 1},
        )
        ind_coadj, matrix_coadj = neighborhood_from_complex(
            domain,
            neighborhood_type="coadj",
            neighborhood_dim={"rank": 1, "via_rank": 0},
        )

        assert isinstance(matrix_adj, csr_matrix)
        assert isinstance(matrix_coadj, csr_matrix)
        assert matrix_adj.shape == (len(ind_adj), len(ind_adj))
        assert matrix_coadj.shape == (len(ind_coadj), len(ind_coadj))
        assert len(ind_adj) == 3
        assert len(ind_coadj) == 3

    def test_neighborhood_from_complex_simplicial_same_rank(self):
        """Testing same-rank neighborhoods for simplicial complexes."""
        domain = tnx.classes.SimplicialComplex([[0, 1, 2]])

        ind_adj, matrix_adj = neighborhood_from_complex(
            domain,
            neighborhood_type="adj",
            neighborhood_dim={"rank": 0},
        )
        ind_coadj, matrix_coadj = neighborhood_from_complex(
            domain,
            neighborhood_type="coadj",
            neighborhood_dim={"rank": 1},
        )

        assert isinstance(matrix_adj, csr_matrix)
        assert isinstance(matrix_coadj, csr_matrix)
        assert matrix_adj.shape == (len(ind_adj), len(ind_adj))
        assert matrix_coadj.shape == (len(ind_coadj), len(ind_coadj))
        assert len(ind_adj) == 3
        assert len(ind_coadj) == 3
        assert set(matrix_adj.data) <= {1}
        assert set(matrix_coadj.data) <= {1}

    def test_neighborhood_from_complex_connection_cell_complex(self):
        """Testing connection graph induced by a single incidence matrix."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert matrix.shape == (6, 6)
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz == 0
        assert set(matrix.data) <= {1}

    def test_neighborhood_from_complex_connection_default_target_rank(self):
        """Testing default target rank for connection neighborhoods."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_default, matrix_default = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0},
        )
        ind_explicit, matrix_explicit = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        assert ind_default == ind_explicit
        assert matrix_default.shape == matrix_explicit.shape
        assert (matrix_default != matrix_explicit).nnz == 0

    def test_neighborhood_from_complex_incidence_aliases(self):
        """Testing incidence aliases for connection neighborhoods."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_inc, matrix_inc = neighborhood_from_complex(
            domain,
            neighborhood_type="inc",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )
        ind_incidence, matrix_incidence = neighborhood_from_complex(
            domain,
            neighborhood_type="incidence",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )
        ind_connection, matrix_connection = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )

        assert ind_inc == ind_incidence == ind_connection
        assert matrix_inc.shape == matrix_incidence.shape == matrix_connection.shape
        assert (matrix_inc != matrix_incidence).nnz == 0
        assert (matrix_inc != matrix_connection).nnz == 0

    def test_neighborhood_from_complex_connection_is_directed_when_requested(self):
        """Testing directed connection graph construction."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={
                "rank": 0,
                "to_rank": 1,
                "symmetric": False,
            },
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz > 0

    def test_neighborhood_from_complex_hasse_cell_complex(self):
        """Testing Hasse graph construction over consecutive ranks."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert matrix.shape == (7, 7)
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz == 0
        assert set(matrix.data) <= {1}

    def test_neighborhood_from_complex_simplicial_hasse(self):
        """Testing Hasse graph construction for simplicial complexes."""
        domain = tnx.classes.SimplicialComplex([[0, 1, 2]])

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert matrix.shape == (7, 7)
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz == 0
        assert set(matrix.data) <= {1}

    def test_neighborhood_from_complex_hasse_default_ranks(self):
        """Testing default Hasse graph rank selection."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_default, matrix_default = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
        )
        ind_explicit, matrix_explicit = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        assert ind_default == ind_explicit
        assert matrix_default.shape == matrix_explicit.shape
        assert (matrix_default != matrix_explicit).nnz == 0

    def test_neighborhood_from_complex_multiple_rank_pairs(self):
        """Testing connection graph from multiple rank-pair matrices."""
        domain = self._small_combinatorial_complex()

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank_pairs": [(0, 1), (1, 2), (0, 2)]},
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert len(ind) == 7
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz == 0

    def test_neighborhood_from_complex_pairs_alias(self):
        """Testing pairs alias for multiple rank-pair matrices."""
        domain = self._small_combinatorial_complex()

        ind_pairs, matrix_pairs = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"pairs": [(0, 1), (1, 2), (0, 2)]},
        )
        ind_rank_pairs, matrix_rank_pairs = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank_pairs": [(0, 1), (1, 2), (0, 2)]},
        )

        assert ind_pairs == ind_rank_pairs
        assert matrix_pairs.shape == matrix_rank_pairs.shape
        assert (matrix_pairs != matrix_rank_pairs).nnz == 0

    def test_neighborhood_from_complex_ranks_to_rank_pairs(self):
        """Testing rank-list conversion to consecutive rank pairs."""
        domain = self._small_combinatorial_complex()

        ind_ranks, matrix_ranks = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )
        ind_pairs, matrix_pairs = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank_pairs": [(0, 1), (1, 2)]},
        )

        assert ind_ranks == ind_pairs
        assert matrix_ranks.shape == matrix_pairs.shape
        assert (matrix_ranks != matrix_pairs).nnz == 0

    def test_neighborhood_from_complex_arbitrary_bij_connection_graph(self):
        """Testing arbitrary B_ij connection graph for combinatorial complexes."""
        domain = self._small_combinatorial_complex()

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 2},
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert len(ind) == 4
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz == 0

    def test_neighborhood_from_complex_combinatorial_directed_connection(self):
        """Testing directed connection graph for combinatorial complexes."""
        domain = self._small_combinatorial_complex()

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={
                "rank": 0,
                "to_rank": 2,
                "symmetric": False,
            },
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert len(ind) == 4
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz > 0

    def test_neighborhood_from_complex_target_rank_alias(self):
        """Testing target_rank alias for connection neighborhoods."""
        domain = self._small_combinatorial_complex()

        ind_target, matrix_target = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "target_rank": 2},
        )
        ind_to_rank, matrix_to_rank = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 2},
        )

        assert ind_target == ind_to_rank
        assert matrix_target.shape == matrix_to_rank.shape
        assert (matrix_target != matrix_to_rank).nnz == 0

    def test_neighborhood_from_complex_via_rank_alias_for_connection(self):
        """Testing via_rank alias for connection neighborhoods."""
        domain = self._small_combinatorial_complex()

        ind_via, matrix_via = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "via_rank": 2},
        )
        ind_to_rank, matrix_to_rank = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 2},
        )

        assert ind_via == ind_to_rank
        assert matrix_via.shape == matrix_to_rank.shape
        assert (matrix_via != matrix_to_rank).nnz == 0

    def test_neighborhood_from_complex_ranked_labels_false(self):
        """Testing unranked labels for cross-rank graphs."""
        domain = self._small_combinatorial_complex()

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={
                "rank": 0,
                "to_rank": 2,
                "ranked_labels": False,
            },
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert len(ind) == 4
        assert all(not isinstance(cell, tuple) for cell in ind)

    def test_neighborhood_from_complex_augmented_hasse_cell_complex(self):
        """Testing augmented Hasse graph with extra same-rank neighborhoods."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_hasse, matrix_hasse = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )
        ind_augmented, matrix_augmented = neighborhood_from_complex(
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

        assert isinstance(matrix_augmented, csr_matrix)
        assert matrix_augmented.shape == (len(ind_augmented), len(ind_augmented))
        assert len(ind_augmented) == len(ind_hasse)
        assert matrix_augmented.shape == matrix_hasse.shape
        assert matrix_augmented.nnz >= matrix_hasse.nnz
        assert (matrix_augmented != matrix_augmented.T).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_without_extra_edges(self):
        """Testing augmented Hasse graph without extra neighborhoods."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_hasse, matrix_hasse = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )
        ind_augmented, matrix_augmented = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )

        assert ind_augmented == ind_hasse
        assert matrix_augmented.shape == matrix_hasse.shape
        assert (matrix_augmented != matrix_hasse).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_combinatorial_complex(self):
        """Testing augmented Hasse graph with arbitrary cross-rank connections."""
        domain = self._small_combinatorial_complex()

        ind, matrix = neighborhood_from_complex(
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

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert len(ind) == 7
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_with_inc_alias(self):
        """Testing augmented Hasse graph with an inc neighborhood."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_hasse, matrix_hasse = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )
        ind_augmented, matrix_augmented = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "inc", "rank": 0, "to_rank": 1},
                ],
            },
        )

        assert len(ind_augmented) == len(ind_hasse)
        assert matrix_augmented.shape == matrix_hasse.shape
        assert matrix_augmented.nnz >= matrix_hasse.nnz
        assert (matrix_augmented != matrix_augmented.T).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_with_incidence_alias(self):
        """Testing augmented Hasse graph with an incidence neighborhood."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_hasse, matrix_hasse = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )
        ind_augmented, matrix_augmented = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "incidence", "rank": 1, "to_rank": 2},
                ],
            },
        )

        assert len(ind_augmented) == len(ind_hasse)
        assert matrix_augmented.shape == matrix_hasse.shape
        assert matrix_augmented.nnz >= matrix_hasse.nnz
        assert (matrix_augmented != matrix_augmented.T).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_with_hasse_neighborhood(self):
        """Testing augmented Hasse graph with a Hasse neighborhood entry."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_hasse, matrix_hasse = neighborhood_from_complex(
            domain,
            neighborhood_type="hasse",
            neighborhood_dim={"ranks": [0, 1, 2]},
        )
        ind_augmented, matrix_augmented = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "neighborhoods": [
                    {"type": "hasse", "ranks": [0, 1]},
                ],
            },
        )

        assert ind_augmented == ind_hasse
        assert matrix_augmented.shape == matrix_hasse.shape
        assert matrix_augmented.nnz >= matrix_hasse.nnz
        assert (matrix_augmented != matrix_augmented.T).nnz == 0

    def test_neighborhood_from_complex_augmented_hasse_directed(self):
        """Testing directed augmented Hasse graph construction."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "ranks": [0, 1, 2],
                "symmetric": False,
            },
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert matrix.shape == (7, 7)
        assert matrix.nnz > 0
        assert (matrix != matrix.T).nnz > 0

    def test_neighborhood_from_complex_augmented_hasse_ranked_labels_false(self):
        """Testing unranked labels for augmented Hasse graphs."""
        domain = self._small_combinatorial_complex()

        ind, matrix = neighborhood_from_complex(
            domain,
            neighborhood_type="augmented_hasse",
            neighborhood_dim={
                "rank_pairs": [(0, 2)],
                "ranked_labels": False,
            },
        )

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (len(ind), len(ind))
        assert len(ind) == 4
        assert all(not isinstance(cell, tuple) for cell in ind)

    def test_neighborhood_from_complex_augmented_hasse_invalid_item(self):
        """Testing invalid augmented neighborhood entries."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(TypeError) as e:
            neighborhood_from_complex(
                domain,
                neighborhood_type="augmented_hasse",
                neighborhood_dim={
                    "ranks": [0, 1, 2],
                    "neighborhoods": ["bad"],
                },
            )

        assert "Each augmented neighborhood must be a mapping." in str(e.value)

    def test_neighborhood_from_complex_augmented_hasse_invalid_type(self):
        """Testing invalid augmented neighborhood types."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(TypeError) as e:
            neighborhood_from_complex(
                domain,
                neighborhood_type="augmented_hasse",
                neighborhood_dim={
                    "ranks": [0, 1, 2],
                    "neighborhoods": [{"type": "bad", "rank": 0}],
                },
            )

        assert "Each augmented neighborhood must define a valid `type`" in str(e.value)

    def test_neighborhood_from_complex_rejects_short_ranks(self):
        """Testing rank-list validation."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(ValueError) as e:
            neighborhood_from_complex(
                domain,
                neighborhood_type="connection",
                neighborhood_dim={"ranks": [0]},
            )

        assert "`ranks` must contain at least two ranks." in str(e.value)

    def test_neighborhood_from_complex_rejects_empty_rank_pairs(self):
        """Testing empty rank-pair validation."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(ValueError) as e:
            neighborhood_from_complex(
                domain,
                neighborhood_type="connection",
                neighborhood_dim={"rank_pairs": []},
            )

        assert "At least one rank pair must be specified." in str(e.value)

    def test_neighborhood_from_complex_rejects_same_rank_pair(self):
        """Testing same-rank pair validation."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(ValueError) as e:
            neighborhood_from_complex(
                domain,
                neighborhood_type="connection",
                neighborhood_dim={"rank_pairs": [(0, 0)]},
            )

        assert "A rank pair must contain two distinct ranks." in str(e.value)

    def test_neighborhood_from_complex_rejects_malformed_rank_pair(self):
        """Testing malformed rank-pair validation."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(ValueError) as e:
            neighborhood_from_complex(
                domain,
                neighborhood_type="connection",
                neighborhood_dim={"rank_pairs": [(0, 1, 2)]},
            )

        assert "Each rank pair must be a two-entry sequence." in str(e.value)

    def test_neighborhood_from_complex_rejects_nonconsecutive_cell_ranks(self):
        """Testing nonconsecutive incidence rejection for cell complexes."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(ValueError) as e:
            neighborhood_from_complex(
                domain,
                neighborhood_type="connection",
                neighborhood_dim={"rank": 0, "to_rank": 2},
            )

        assert "Non-combinatorial complexes support Hasse incidence only" in str(e.value)

    @staticmethod
    def _small_combinatorial_complex():
        """Create a small combinatorial complex with ranks 0, 1, and 2.

        Returns
        -------
        toponetx.classes.CombinatorialComplex
            A small combinatorial complex.
        """
        domain = tnx.classes.CombinatorialComplex()
        domain.add_cell([0], rank=0)
        domain.add_cell([1], rank=0)
        domain.add_cell([2], rank=0)
        domain.add_cell([0, 1], rank=1)
        domain.add_cell([1, 2], rank=1)
        domain.add_cell([0, 2], rank=1)
        domain.add_cell([0, 1, 2], rank=2)
        return domain
