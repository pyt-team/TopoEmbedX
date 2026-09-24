"""Testing the neighborhood module."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import toponetx as tnx
from scipy.sparse import csr_matrix

import topoembedx.neighborhood as neighborhood_module
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

    def test_incidence_between_ranks_rejects_equal_ranks(self):
        """Testing direct incidence rejection for equal ranks."""
        incidence_between_ranks = neighborhood_module.__dict__[
            "_incidence_between_ranks"
        ]
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        with pytest.raises(ValueError) as e:
            incidence_between_ranks(domain, 0, 0)

        assert "Incidence rank pairs must contain two distinct ranks." in str(e.value)

    def test_incidence_between_ranks_handles_reversed_ranks(self):
        """Testing direct incidence handling for reversed ranks."""
        incidence_between_ranks = neighborhood_module.__dict__[
            "_incidence_between_ranks"
        ]
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        row_cells, col_cells, matrix = incidence_between_ranks(domain, 1, 0)

        assert len(row_cells) == 3
        assert len(col_cells) == 3
        assert matrix.shape == (3, 3)
        assert matrix.nnz > 0

    def test_incidence_between_ranks_uses_noncombinatorial_fallback_candidate(self):
        """Testing incidence fallback for non-combinatorial APIs."""
        incidence_between_ranks = neighborhood_module.__dict__[
            "_incidence_between_ranks"
        ]
        incidence_matrix = Mock(
            side_effect=[
                TypeError("unsupported signature"),
                ({"low": 0}, {"high": 0}, csr_matrix([[1]])),
            ]
        )
        domain = SimpleNamespace(incidence_matrix=incidence_matrix)

        row_cells, col_cells, matrix = incidence_between_ranks(domain, 0, 1)

        assert incidence_matrix.call_count == 2
        assert row_cells == ["low"]
        assert col_cells == ["high"]
        assert matrix.shape == (1, 1)
        assert matrix.nnz == 1

    def test_incidence_between_ranks_raises_when_all_candidates_fail(self):
        """Testing incidence failure when all API signatures fail."""
        incidence_between_ranks = neighborhood_module.__dict__[
            "_incidence_between_ranks"
        ]
        incidence_matrix = Mock(side_effect=TypeError("unsupported signature"))
        domain = SimpleNamespace(incidence_matrix=incidence_matrix)

        with pytest.raises(TypeError) as e:
            incidence_between_ranks(domain, 0, 1)

        assert "Unable to compute an incidence matrix" in str(e.value)
        assert incidence_matrix.call_count == 4

    def test_unpack_incidence_result_rejects_non_tuple(self):
        """Testing incidence unpacking rejects non-tuple results."""
        unpack_incidence_result = neighborhood_module.__dict__[
            "_unpack_incidence_result"
        ]

        with pytest.raises(TypeError) as e:
            unpack_incidence_result(csr_matrix([[1]]))

        assert "Expected TopoNetX to return a tuple." in str(e.value)

    def test_unpack_incidence_result_accepts_nested_index_pair(self):
        """Testing incidence unpacking accepts nested index pairs."""
        unpack_incidence_result = neighborhood_module.__dict__[
            "_unpack_incidence_result"
        ]
        matrix = csr_matrix([[2, 0], [0, 3]])

        row_cells, col_cells, unpacked = unpack_incidence_result(
            (({"row0": 0, "row1": 1}, {"col0": 0, "col1": 1}), matrix)
        )

        assert row_cells == ["row0", "row1"]
        assert col_cells == ["col0", "col1"]
        assert unpacked.shape == (2, 2)
        assert set(unpacked.data) <= {1}

    def test_unpack_incidence_result_rejects_malformed_tuple(self):
        """Testing incidence unpacking rejects malformed tuple results."""
        unpack_incidence_result = neighborhood_module.__dict__[
            "_unpack_incidence_result"
        ]

        with pytest.raises(TypeError) as e:
            unpack_incidence_result(({"row": 0}, csr_matrix([[1]])))

        assert "Expected incidence_matrix" in str(e.value)

    def test_orient_incidence_result_transposes_reversed_orientation(self):
        """Testing incidence orientation transposes reversed incidence matrices."""
        orient_incidence_result = neighborhood_module.__dict__[
            "_orient_incidence_result"
        ]
        cells_of_rank = neighborhood_module.__dict__["_cells_of_rank"]
        domain = tnx.classes.CellComplex([[0, 1, 2]])
        low_cells = cells_of_rank(domain, 1)
        high_cells = cells_of_rank(domain, 2)
        matrix = csr_matrix([[1, 1, 1]])

        row_cells, col_cells, oriented = orient_incidence_result(
            domain,
            1,
            2,
            high_cells,
            low_cells,
            matrix,
        )

        assert row_cells == low_cells
        assert col_cells == high_cells
        assert oriented.shape == (len(low_cells), len(high_cells))

    def test_orient_incidence_result_returns_unknown_orientation(self):
        """Testing incidence orientation keeps unknown orientation unchanged."""
        orient_incidence_result = neighborhood_module.__dict__[
            "_orient_incidence_result"
        ]
        domain = tnx.classes.CellComplex([[0, 1, 2]])
        matrix = csr_matrix([[1]])

        row_cells, col_cells, oriented = orient_incidence_result(
            domain,
            0,
            1,
            ["unknown-row"],
            ["unknown-col"],
            matrix,
        )

        assert row_cells == ["unknown-row"]
        assert col_cells == ["unknown-col"]
        assert oriented is matrix

    def test_orient_incidence_result_returns_when_rank_cells_are_missing(self):
        """Testing incidence orientation returns when rank cells are unavailable."""
        orient_incidence_result = neighborhood_module.__dict__[
            "_orient_incidence_result"
        ]
        matrix = csr_matrix([[1]])

        row_cells, col_cells, oriented = orient_incidence_result(
            SimpleNamespace(),
            0,
            1,
            ["row"],
            ["col"],
            matrix,
        )

        assert row_cells == ["row"]
        assert col_cells == ["col"]
        assert oriented is matrix

    def test_same_rank_neighborhood_rejects_unsupported_complex(self):
        """Testing same-rank helper rejects unsupported complex types."""
        same_rank_neighborhood = neighborhood_module.__dict__["_same_rank_neighborhood"]

        with pytest.raises(TypeError) as e:
            same_rank_neighborhood(
                SimpleNamespace(),
                "adj",
                {"rank": 0},
            )

        assert "Unsupported complex type." in str(e.value)

    def test_domain_dimension_handles_missing_callable_and_value_cases(self):
        """Testing dimension helper handles all exposed dimension forms."""
        domain_dimension = neighborhood_module.__dict__["_domain_dimension"]

        assert domain_dimension(SimpleNamespace()) is None
        assert domain_dimension(SimpleNamespace(dim=Mock(return_value="3"))) == 3
        assert (
            domain_dimension(
                SimpleNamespace(
                    dim=Mock(side_effect=TypeError("unsupported")),
                    dimension=2,
                )
            )
            == 2
        )

    def test_cells_of_rank_handles_missing_and_keyword_skeleton_cases(self):
        """Testing cell-rank helper handles missing and keyword skeleton APIs."""
        cells_of_rank = neighborhood_module.__dict__["_cells_of_rank"]

        assert cells_of_rank(SimpleNamespace(), 0) == []

        skeleton = Mock(
            side_effect=[
                TypeError("use keyword"),
                [{0, 1}],
            ]
        )
        domain = SimpleNamespace(skeleton=skeleton)

        assert cells_of_rank(domain, 1) == [frozenset({0, 1})]
        assert skeleton.call_count == 2

    def test_ordered_cells_handles_mapping_and_unhashable_iterables(self):
        """Testing ordered-cell helper handles mappings and unhashable cells."""
        ordered_cells = neighborhood_module.__dict__["_ordered_cells"]

        assert ordered_cells({"b": 1, "a": 0}) == ["a", "b"]
        assert ordered_cells([{0, 1}, [2, 3], SimpleNamespace()]) == [
            frozenset({0, 1}),
            (2, 3),
            "namespace()",
        ]

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
