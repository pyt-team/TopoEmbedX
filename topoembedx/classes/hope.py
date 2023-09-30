"""Higher Order Laplacian Positional Encoder (HOPE) class."""
import networkx as nx
import numpy as np
from scipy import sparse

from topoembedx.neighborhood import neighborhood_from_complex


class HOPE:
    """Higher Order Laplacian Positional Encoder (HOPE) class.

    Parameters
    ----------
    dimensions : int, optional
        Dimensionality of embedding. Defaults to 3.
    """

    def __init__(self, dimensions: int = 2):

        self.A = []
        self.ind = []
        self.dimensions = dimensions

        self._embedding = []

    @staticmethod
    def _laplacian_pe(A, k: int, return_eigenval: bool = False):
        """
        Compute Laplacian Positional Encodings (PE) for a given adjacency matrix.

        Parameters
        ----------
        A : numpy.ndarray, shape (n, n)
            Adjacency matrix representing the graph structure.
        k : int
            Number of eigenvectors to consider for Laplacian PE.
        return_eigenval : bool, optional (default=False)
            Whether to return the eigenvalues along with PE.

        Returns
        -------
        numpy.ndarray or tuple
            Laplacian Positional Encodings computed from the Laplacian eigenvectors.
            If return_eigenval is True, returns a tuple (PE, eigenvalues).

        Explanation
        -----------
        This function computes Laplacian Positional Encodings (PE) based on the input
        adjacency matrix and the desired number of eigenvectors. The Laplacian PE is a
        representation of the graph structure obtained from the Laplacian eigenvectors.

        The Laplacian matrix L is computed using the input adjacency matrix A and then
        its eigenvectors are calculated. The k smallest non-trivial eigenvectors are
        selected to form the Laplacian PE.

        If the number of eigenvectors (k) is less than or equal to the number of nodes (n),
        the remaining dimensions are filled with zeros. The signs of the eigenvectors are
        randomly flipped to enhance representational capacity.

        Finally, the Laplacian positional encodings are returned for further usage. If
        return_eigenval is True, eigenvalues are also returned alongside the positional encodings.
        """
        n = A.shape[0]
        # Compute the degree matrix D^-0.5
        D = sparse.diags(np.squeeze(np.asarray(np.power(np.sum(A, axis=1), -0.5))))

        # Compute the Laplacian matrix L = I - D^-0.5 * A * D^-0.5
        L = np.eye(A.shape[0]) - D @ A @ D

        # Compute the eigenvectors of L
        EigVal, EigVec = np.linalg.eig(L)

        # Select the k smallest non-trivial eigenvectors
        max_freqs = min(n - 1, k)
        kpartition_indices = np.argpartition(EigVal, max_freqs)[: max_freqs + 1]
        topk_eigvals = EigVal[kpartition_indices]
        topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
        topk_EigVec = EigVec[:, topk_indices]

        # Randomly flip signs of the eigenvectors
        rand_sign = 2 * (np.random.rand(max_freqs) > 0.5) - 1.0
        PE = np.multiply(rand_sign, topk_EigVec.astype(np.float32))

        if n <= k:
            temp_EigVec = np.zeros((n, k - n + 1), dtype=np.float32)
            PE = np.concatenate((PE, temp_EigVec), axis=1)
            temp_EigVal = np.full(k - n + 1, np.nan, dtype=np.float32)
            eigvals = np.concatenate((topk_eigvals, temp_EigVal), axis=0)
        else:
            eigvals = topk_eigvals

        if return_eigenval:
            return np.array(PE), eigvals

        # Return the Laplacian positional encodings
        return np.array(PE)

    def fit(
        self,
        complex,
        neighborhood_type="adj",
        neighborhood_dim={"rank": 0, "to_rank": -1},
    ):
        """Fit a Higher Order Geometric Laplacian EigenMaps model.

        Parameters
        ----------
        complex : TopoNetX object
            A complex object. The complex object can be one of the following:
            - CellComplex
            - CombinatorialComplex
            - ColoredHyperGraph
            - SimplicialComplex
            - PathComplex
        neighborhood_type : str
            The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
        neighborhood_dim : dict
            The dimensions of the neighborhood to use. If `neighborhood_type` is "adj", the dimension is
            `neighborhood_dim['rank']`. If `neighborhood_type` is "coadj", the dimension is `neighborhood_dim['rank']`
            and `neighborhood_dim['to_rank']` specifies the dimension of the ambient space.

        Notes
        -----
        Here, neighborhood_dim={"rank": 1, "to_rank": -1} specifies the dimension for
        which the cell embeddings are going to be computed.
        rank=1 means that the embeddings will be computed for the first dimension.
        The integer 'to_rank' is ignored and only considered
        when the input complex is a combinatorial complex.

        Examples
        --------
        import toponetx as tnx
        from topoembedx import HOPE
        ccc = tnx.classes.CombinatorialComplex()
        ccc.add_cell([2,5],rank=1)
        ccc.add_cell([2,4],rank=1)
        ccc.add_cell([7,8],rank=1)
        ccc.add_cell([6,8],rank=1)
        ccc.add_cell([2,4,5],rank=3)
        ccc.add_cell([6,7,8],rank=3)

        model = HOPE()
        model.fit(ccc, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank" :3})
        em=model.get_embedding(get_dict=True)


        Returns
        -------
        None
        """
        self.ind, self.A = neighborhood_from_complex(
            complex, neighborhood_type, neighborhood_dim
        )

        self._embedding = HOPE._laplacian_pe(self.A, self.dimensions)

    def get_embedding(self, get_dict=False):
        """Get embedding.

        Parameters
        ----------
        get_dict : bool, optional
            Whether to return a dictionary. Defaults to False.

        Returns
        -------
        dict or numpy.ndarray
            Embedding.
        """
        emb = self._embedding
        if get_dict:
            return dict(zip(self.ind, emb))
        else:
            return emb