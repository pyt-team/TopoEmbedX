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

    def test_neighborhood_from_complex_reversed_connection_rank_pair(self):
        """Testing reversed rank-pair normalization for connection graphs."""
        domain = tnx.classes.CellComplex([[0, 1, 2]])

        ind_forward, matrix_forward = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 1},
        )
        ind_reversed, matrix_reversed = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 1, "to_rank": 0},
        )

        assert ind_forward == ind_reversed
        assert matrix_forward.shape == matrix_reversed.shape
        assert (matrix_forward != matrix_reversed).nnz == 0

    def test_neighborhood_from_complex_reversed_rank_pairs_list(self):
        """Testing reversed rank pairs inside rank-pair lists."""
        domain = self._small_combinatorial_complex()

        ind_forward, matrix_forward = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank_pairs": [(0, 1), (0, 2)]},
        )
        ind_reversed, matrix_reversed = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank_pairs": [(1, 0), (2, 0)]},
        )

        assert ind_forward == ind_reversed
        assert matrix_forward.shape == matrix_reversed.shape
        assert (matrix_forward != matrix_reversed).nnz == 0

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

    def test_neighborhood_from_complex_augmented_hasse_with_augmented_entry(self):
        """Testing augmented Hasse graph with an augmented Hasse entry."""
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
                    {"type": "augmented_hasse", "ranks": [1, 2]},
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

    def test_neighborhood_from_complex_combinatorial_reversed_bij_connection(self):
        """Testing reversed arbitrary B_ij connection for combinatorial complexes."""
        domain = self._small_combinatorial_complex()

        ind_forward, matrix_forward = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 0, "to_rank": 2},
        )
        ind_reversed, matrix_reversed = neighborhood_from_complex(
            domain,
            neighborhood_type="connection",
            neighborhood_dim={"rank": 2, "to_rank": 0},
        )

        assert ind_forward == ind_reversed
        assert matrix_forward.shape == matrix_reversed.shape
        assert (matrix_forward != matrix_reversed).nnz == 0

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
