# Existing behavior.
ind, A = neighborhood_from_complex(
    domain,
    neighborhood_type="coadj",
    neighborhood_dim={"rank": 1, "via_rank": 2},
)

# One B_ij connection graph, e.g. B_02 for a CC/CHG.
ind, A = neighborhood_from_complex(
    domain,
    neighborhood_type="connection",
    neighborhood_dim={"rank": 0, "to_rank": 2},
)

# Multiple B_ij graphs in one matrix.
ind, A = neighborhood_from_complex(
    domain,
    neighborhood_type="connection",
    neighborhood_dim={"rank_pairs": [(0, 1), (1, 2), (0, 2)]},
)

# Full Hasse graph over consecutive ranks.
ind, A = neighborhood_from_complex(
    domain,
    neighborhood_type="hasse",
    neighborhood_dim={"ranks": [0, 1, 2]},
)

# Augmented Hasse graph: Hasse + same-rank and cross-rank relations.
ind, A = neighborhood_from_complex(
    domain,
    neighborhood_type="augmented_hasse",
    neighborhood_dim={
        "ranks": [0, 1, 2],
        "neighborhoods": [
            {"type": "adj", "rank": 0, "via_rank": 1},
            {"type": "coadj", "rank": 1, "via_rank": 0},
            {"type": "connection", "rank": 0, "to_rank": 2},
        ],
    },
)
