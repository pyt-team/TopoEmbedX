"""Testing the neighborhood module."""

import pytest
import toponetx as tnx
from scipy.sparse import csr_matrix

import topoembedx as tex


class TestNeighborhood:
    """Test the neighborhood module of TopoEmbedX."""

    # ------------------------------------------------------------------
    # Original tests (unchanged)
    # ------------------------------------------------------------------
    def test_neighborhood_from_complex_raise_error(self):
        """Testing if right assertion is raised for incorrect type."""
        with pytest.raises(TypeError) as e:
            tex.neighborhood.neighborhood_from_complex(1)

        # NOTE: this requires the error message string in the implementation
        # to match exactly.
        assert (
            str(e.value)
            == """Input Complex can only be a SimplicialComplex, CellComplex, PathComplex ColoredHyperGraph or CombinatorialComplex."""
        )

    def test_neighborhood_from_complex_matrix_dimension_cell_complex(self):
        """Testing the matrix dimensions for the adjacency and coadjacency matrices.

        This checks that:
        - adjacency on the default rank (0) returns a square matrix,
        - the size matches the number of 0-cells (nodes),
        - coadjacency has the same behavior.
        """
        # Testing for the case of Cell Complex
        cc1 = tnx.classes.CellComplex(
            [[0, 1, 2, 3], [1, 2, 3, 4], [1, 3, 4, 5, 6, 7, 8]]
        )

        cc2 = tnx.classes.CellComplex([[0, 1, 2], [1, 2, 3]])

        # Default: adjacency on rank=0 (nodes)
        ind, A = tex.neighborhood.neighborhood_from_complex(cc1)
        assert A.todense().shape == (9, 9)
        assert len(ind) == 9

        ind, A = tex.neighborhood.neighborhood_from_complex(cc2)
        assert A.todense().shape == (4, 4)
        assert len(ind) == 4

        # Coadjacency on rank=0 (default behavior in neighborhood_from_complex)
        ind, A = tex.neighborhood.neighborhood_from_complex(
            cc1, neighborhood_type="coadj"
        )
        assert A.todense().shape == (9, 9)
        assert len(ind) == 9

        ind, A = tex.neighborhood.neighborhood_from_complex(
            cc2, neighborhood_type="coadj"
        )
        assert A.todense().shape == (4, 4)
        assert len(ind) == 4

    # ------------------------------------------------------------------
    # New tests: SimplicialComplex adjacency / coadjacency
    # ------------------------------------------------------------------
    def test_neighborhood_from_complex_simplicial_adj_coadj(self):
        """Adjacency and coadjacency matrices on a simplicial complex.

        We build a small simplicial complex with two triangles that share
        an edge, and verify that:

        - adjacency on rank=1 (edges) is square and has size = #edges,
        - coadjacency on rank=2 (faces) is square and has size = #faces.
        """
        sc = tnx.SimplicialComplex([[0, 1, 2], [1, 2, 3]])

        # Adjacency on rank=1 (edges)
        ind_adj, A_adj = tex.neighborhood.neighborhood_from_complex(
            sc,
            neighborhood_type="adj",
            neighborhood_dim={"rank": 1, "via_rank": -1},
        )
        assert isinstance(A_adj, csr_matrix)
        assert A_adj.shape[0] == A_adj.shape[1]
        assert len(ind_adj) == A_adj.shape[0]

        # Coadjacency on rank=2 (faces)
        ind_coadj, A_coadj = tex.neighborhood.neighborhood_from_complex(
            sc,
            neighborhood_type="coadj",
            neighborhood_dim={"rank": 2, "via_rank": -1},
        )
        assert isinstance(A_coadj, csr_matrix)
        assert A_coadj.shape[0] == A_coadj.shape[1]
        assert len(ind_coadj) == A_coadj.shape[0]

    # ------------------------------------------------------------------
    # New tests: boundary / coboundary via incidence_matrix (Hasse graph)
    # ------------------------------------------------------------------
    def test_neighborhood_from_complex_boundary_coboundary_simplicial(self):
        """Boundary and coboundary neighborhoods on a simplicial complex.

        We use a single 2-simplex [0,1,2]. For rank=1:

        - `incidence_matrix(rank=1)` relates vertices (rank 0) and edges (rank 1),
        - the Hasse graph has nodes = vertices ∪ edges,
        - boundary and coboundary should return the same undirected adjacency.
        """
        sc = tnx.SimplicialComplex([[0, 1, 2]])

        # Boundary neighborhood (Hasse graph from incidence between rank 1 and 0)
        ind_b, A_b = tex.neighborhood.neighborhood_from_complex(
            sc,
            neighborhood_type="boundary",
            neighborhood_dim={"rank": 1},
        )
        assert isinstance(A_b, csr_matrix)
        assert A_b.shape[0] == A_b.shape[1]
        assert len(ind_b) == A_b.shape[0]
        # Should have at least some non-zero entries (edges between ranks 0 and 1)
        assert A_b.nnz > 0

        # Coboundary neighborhood: same Hasse graph in this undirected setting
        ind_cb, A_cb = tex.neighborhood.neighborhood_from_complex(
            sc,
            neighborhood_type="coboundary",
            neighborhood_dim={"rank": 1},
        )
        assert isinstance(A_cb, csr_matrix)
        assert A_cb.shape == A_b.shape
        assert len(ind_cb) == len(ind_b)
        # Matrices should be identical
        assert (A_cb != A_b).nnz == 0

    # ------------------------------------------------------------------
    # New tests: CombinatorialComplex adjacency / coadjacency / boundary
    # ------------------------------------------------------------------
    def test_neighborhood_from_complex_combinatorial_complex(self):
        """Adjacency, coadjacency, and boundary neighborhoods on a combinatorial complex.

        We build a small combinatorial complex with:
        - rank 0: vertices {0,1,2,3}
        - rank 1: edges of a 4-cycle
        - rank 2: two faces [0,1,2] and [0,2,3]

        We verify:
        - adjacency on rank=1 via rank=2 is square,
        - coadjacency on rank=1 via rank=0 is square,
        - boundary neighborhood at rank=2 (if incidence_matrix is implemented)
          produces a square Hasse adjacency over ranks 1 and 2.
        """
        cc = tnx.CombinatorialComplex()
        # vertices
        for v in [0, 1, 2, 3]:
            cc.add_cell([v], rank=0)
        # edges (4-cycle)
        for e in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cc.add_cell(list(e), rank=1)
        # 2-cells
        cc.add_cell([0, 1, 2], rank=2)
        cc.add_cell([0, 2, 3], rank=2)

        # Rank-1 adjacency via rank-2
        ind_adj, A_adj = tex.neighborhood.neighborhood_from_complex(
            cc,
            neighborhood_type="adj",
            neighborhood_dim={"rank": 1, "via_rank": 2},
        )
        assert isinstance(A_adj, csr_matrix)
        assert A_adj.shape[0] == A_adj.shape[1]
        assert len(ind_adj) == A_adj.shape[0]

        # Rank-1 coadjacency via rank-0
        ind_coadj, A_coadj = tex.neighborhood.neighborhood_from_complex(
            cc,
            neighborhood_type="coadj",
            neighborhood_dim={"rank": 1, "via_rank": 0},
        )
        assert isinstance(A_coadj, csr_matrix)
        assert A_coadj.shape[0] == A_coadj.shape[1]
        assert len(ind_coadj) == A_coadj.shape[0]

        # Boundary neighborhood at rank=2: incidence between rank 2 and rank 1
        # Not all TopoNetX versions may have incidence_matrix for CC, so we guard it.
        try:
            ind_b, A_b = tex.neighborhood.neighborhood_from_complex(
                cc,
                neighborhood_type="boundary",
                neighborhood_dim={"rank": 2},
            )
        except TypeError:
            # If incidence_matrix is not implemented for CombinatorialComplex,
            # we skip the boundary check gracefully.
            pytest.skip("CombinatorialComplex.incidence_matrix not available in this TopoNetX version.")
        else:
            assert isinstance(A_b, csr_matrix)
            assert A_b.shape[0] == A_b.shape[1]
            assert len(ind_b) == A_b.shape[0]
            assert A_b.nnz > 0
