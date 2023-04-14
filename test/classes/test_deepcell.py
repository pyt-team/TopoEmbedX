import unittest

import numpy as np
import toponetx as tnx

from topoembedx.classes.deepcell import DeepCell


class TestDeepCell(unittest.TestCase):
    def test_DeepCell(self):
        # Create a small graph
        cx = tnx.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

        # Create a DeepCell object
        dc = DeepCell(walk_number=5, walk_length=10, dimensions=2)

        # Fit the DeepCell object to the graph
        dc.fit(cx, neighborhood_type="adj", neighborhood_dim={"r": 1, "k": -1})

        # Check that the shape of the embedding is correct
        assert dc.get_embedding().shape == (8, 2)

        # Check that the shape of the embedding dictionary is correct
        ind, _ = dc.get_embedding(get_dic=True)
        assert (len(ind)) == 8

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(dc.get_embedding()[0], dc.get_embedding()[1])


if __name__ == "__main__":
    unittest.main()
