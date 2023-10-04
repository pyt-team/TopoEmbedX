"""Test CellDiff2Vec class."""

import pytest
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
