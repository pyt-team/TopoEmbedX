"""Generate random walks on a graph or network.

To generate node embeddings using Word2Vec, you can first use the
random_walk function to generate random walks on your complex.
Then, you can use the generated random walks as input to the Word2Vec
algorithm to learn node embeddings.

Examples
--------
Here is an example of how you could use the random_walk function
and Word2Vec to generate cell embeddings:

>>> # Import the necessary modules
>>> from gensim.models import Word2Vec
>>> # Generate random walks on your graph using the random_walk function
>>> random_walks = random_walk(length=10, num_walks=100, states=nodes, transition_matrix=transition_matrix)
>>> # Train a Word2Vec model on the generated random walks
>>> model = Word2Vec(random_walks, size=128, window=5, min_count=0, sg=1)
>>> # Use the trained model to generate node embeddings
>>> cell_embeddings = model.wv

In the example above, `nodes` is a list of the nodes in your complex,
`transition_matrix` is the transition matrix of your complex, and `cell_embeddings`
is a dictionary that maps each node in your complex to its corresponding embedding.
"""
from typing import TypeVar

import numpy as np
from pyrandwalk import RandomWalk
from sklearn.preprocessing import normalize

T = TypeVar("T")


def transition_from_adjacency(
    A: np.ndarray, sub_sampling: float = 0.1, self_loop: bool = True
) -> np.ndarray:
    """Generate transition matrix from an adjacency matrix.

    This function generates a transition matrix from an adjacency matrix
    using the following steps:

    1. Add self-loop to the adjaency matrix if self_loop is set to True
    2. Compute the degree matrix
    3. Compute the transition matrix by taking the dot product of the inverse of
       the degree matrix and the adjacency matrix

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix.
    sub_sampling : float, default=0.1
        The rate of subsampling.
    self_loop : bool, default=True
        A flag indicating whether to add self-loop to the adjacency matrix.

    Returns
    -------
    numpy.ndarray
        The transition matrix.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])
    >>> transition_from_adjacency(A)
    array([[0.33333333, 0.33333333, 0.33333333, 0.        ],
           [0.33333333, 0.33333333, 0.33333333, 0.        ],
           [0.25      , 0.25      , 0.25      , 0.25      ],
           [0.        , 0.        , 0.5       , 0.5       ]])
    """

    def _transition_from_adjacency(A: np.ndarray):
        A = A + np.eye(A.shape[0])
        # let's evaluate the degree matrix D
        D = np.diag(np.sum(A, axis=0))
        # ...and the transition matrix T
        return np.dot(np.linalg.inv(D), A)

    def _weight_node(A: np.ndarray, sub_sampling: float = sub_sampling):
        z = np.array(np.abs(A).sum(1)) + 1
        weight = 1 / (z**sub_sampling)
        return weight.T

    def get_normalized_adjacency(
        A: np.ndarray, sub_sampling: float = sub_sampling
    ) -> np.ndarray:
        """Get normalized adjacency matrix.

        Parameters
        ----------
        A : numpy.ndarray
            The adjacency matrix.
        sub_sampling : float, optional
            The rate of subsampling, by default 0.1.

        Returns
        -------
        numpy.ndarray
            The normalized adjacency matrix.
        """
        if sub_sampling != 0:
            print(_weight_node(A, sub_sampling))
            D_inv = np.diag(_weight_node(A, sub_sampling))
            A = A.dot(D_inv)
        normalize(A, norm="l1", axis=1, copy=False)
        return A

    if self_loop:
        return _transition_from_adjacency(A)
    return get_normalized_adjacency(A, sub_sampling)


def random_walk(
    length: int, num_walks: int, states: list[T], transition_matrix: np.ndarray
) -> list[list[T]]:
    """Generate random walks on a graph or network.

    This function generates random walks of a given length on a given complex.
    The length and number of walks can be specified, as well as the cells (states)
    and transition matrix.

    Parameters
    ----------
    length : int
        The length of each random walk.
    num_walks : int
        The number of random walks to generate.
    states : list
        The nodes in the complex.
    transition_matrix : numpy.ndarray
        The transition matrix of the graph or network.

    Returns
    -------
    list of list
        The generated random walks.

    Examples
    --------
    >>> import numpy as np
    >>> transition_matrix = np.array(
    ...     [
    ...         [0.0, 1.0, 0.0, 0.0],
    ...         [0.5, 0.0, 0.5, 0.0],
    ...         [0.0, 0.5, 0.0, 0.5],
    ...         [0.0, 0.0, 1.0, 0.0],
    ...     ]
    ... )
    >>> states = ["A", "B", "C", "D"]
    >>> walks = random_walk(
    ...     length=3, num_walks=2, states=states, transition_matrix=transition_matrix
    ... )
    >>> print(walks)
    [['B', 'C', 'D'], ['B', 'C', 'B']]
    """
    rw = RandomWalk(states, transition_matrix)
    walks = []
    for _ in range(num_walks):
        states, _ = rw.run(length - 1)
        walks.append(states)
    return walks
