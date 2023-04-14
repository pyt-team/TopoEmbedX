import unittest

import numpy as np
import toponetx as tnx

from topoembedx.classes.cell_diff2vec import CellDiff2Vec


class TestDiff2Vec(unittest.TestCase):
    def test_Diff2Vec(self):
        # Create a small graph
        sc = tnx.SimplicialComplex()
        sc.add_simplex([1, 2, 3, 4])
        sc.add_simplex([5, 6, 7, 8])

        # Create a CellDiff2Vec object
        dc = CellDiff2Vec(dimensions=5)

        # Fit the CellDiff2Vec object to the graph and get embedding for 2 simpliex (using adjacency matrix A0)
        dc.fit(sc, neighborhood_type="adj", neighborhood_dim={"r": 2, "k": -1})

        # Check that the shape of the embedding is correct
        assert dc.get_embedding().shape == (len(sc.skeleton(2)), 5)

        # Check that the shape of the embedding dictionary is correct
        ind = dc.get_embedding(get_dic=True)
        assert (len(ind)) == len(sc.edges)

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(dc.get_embedding()[0], dc.get_embedding()[1])


if __name__ == "__main__":
    unittest.main()
