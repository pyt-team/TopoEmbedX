"""Functions for computing neighborhoods of a complex."""

from toponetx.classes import (
    CellComplex,
    ColoredHyperGraph,
    CombinatorialComplex,
    PathComplex,
    SimplicialComplex,
)


def neighborhood_from_complex(
    complex, neighborhood_type="adj", neighborhood_dim={"dim": 0, "codim": -1}
):
    """Compute the neighborhood of a complex.

    This function returns the indices and matrix for the neighborhood specified
    by `neighborhood_type`
    and `neighborhood_dim` for the input complex `complex`.

    Parameters
    ----------
    complex : CellComplex, CombinatorialComplex, SimplicialComplex, ColoredHyperGraph, PathComplex
        The complex to compute the neighborhood for.
    neighborhood_type : str
        The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
    neighborhood_dim : dict
        The integer parmaters needed to specify the neighborhood of the cells to generate the embedding.
        In TopoNetX  (co)adjacency neighborhood matrices are specified via one or two parameters.
        - For Cell/Simplicial/Path complexes (co)adjacency matrix is specified by a single parameter, this is precisely
        neighborhood_dim["dim"]
        - For Combinatorial/ColoredHyperGraph the (co)adjacency matrix is specified by a single parameter, this is precisely
        neighborhood_dim["dim"] and neighborhood_dim["codim"]

    Notes
    -----
        here neighborhood_dim={"dim": 1, "codim": -1} specifies the dimension for
        which the cell embeddings are going to be computed.
        "dim": 1 means that the embeddings will be computed for the first dimension.
        The integer "codim": -1 is ignored when the input is cell/simplicial complex
        and  must be specified when the input complex is a combinatorial complex or
        colored hypergraph.

    Returns
    -------
    ind : list
        A list of the indices for the nodes in the neighborhood.
    A : ndarray
        The matrix representing the neighborhood.

    Raises
    ------
    ValueError
        If the input `complex` is not a SimplicialComplex, CellComplex or CombinatorialComplex
    """
    if isinstance(complex, (SimplicialComplex, CellComplex, PathComplex)):
        if neighborhood_type == "adj":
            ind, A = complex.adjacency_matrix(neighborhood_dim["dim"], index=True)

        else:
            ind, A = complex.coadjacency_matrix(neighborhood_dim["dim"], index=True)
    elif isinstance(complex, (CombinatorialComplex, ColoredHyperGraph)):
        if neighborhood_type == "adj":
            ind, A = complex.adjacency_matrix(
                neighborhood_dim["dim"], neighborhood_dim["codim"], index=True
            )
        else:
            ind, A = complex.coadjacency_matrix(
                neighborhood_dim["codim"], neighborhood_dim["codim"], index=True
            )
    else:
        raise TypeError(
            """Input Complex can only be a SimplicialComplex, CellComplex, PathComplex ColoredHyperGraph or CombinatorialComplex."""
        )

    return ind, A
