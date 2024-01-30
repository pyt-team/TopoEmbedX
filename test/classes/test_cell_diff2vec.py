"""Test CellDiff2Vec class."""

import toponetx as tnx

from topoembedx.classes.cell_diff2vec import CellDiff2Vec


class TestDiff2Vec:
    """Test Diff2Vec class."""

    def test_init(self):
        """Test get_embedding."""
        # Create a small graph
        sc = tnx.classes.SimplicialComplex()
        sc.add_simplex([0, 1])

        # Create a CellDiff2Vec object
        _ = CellDiff2Vec(dimensions=2)

    def test_fit(self):
        """Test the fit method."""
        # Create a larger graph
        cx_large = tnx.classes.CellComplex(
            [[i, i + 1, i + 2, i + 3] for i in range(0, 100, 3)], ranks=2
        )

        # Create a CellDiff2Vec object
        model = CellDiff2Vec(dimensions=2)

        # Fit the CellDiff2Vec object to the graph and get embedding for nodes
        model.fit(cx_large)
