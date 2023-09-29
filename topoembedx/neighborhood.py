"""Functions for computing neighborhoods of a complex."""

from toponetx.classes import (
    CellComplex,
    ColoredHyperGraph,
    CombinatorialComplex,
    PathComplex,
    SimplicialComplex,
)


def neighborhood_from_complex(
    complex, neighborhood_type="adj", neighborhood_dim={"rank": 0, "via_rank": -1}
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
        neighborhood_dim["rank"]
        - For Combinatorial/ColoredHyperGraph the (co)adjacency matrix is specified by a single parameter, this is precisely
        neighborhood_dim["rank"] and neighborhood_dim["via_rank"]

    Notes
    -----
        here neighborhood_dim={"rank": 1, "via_rank": -1} specifies the dimension for
        which the cell embeddings are going to be computed.
        "rank": 1 means that the embeddings will be computed for the first dimension.
        The integer "via_rank": -1 is ignored when the input is cell/simplicial complex
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
        If the input `complex` is not a SimplicialComplex, CellComplex, PathComplex ColoredHyperGraph or CombinatorialComplex.
    """
    if isinstance(complex, (SimplicialComplex, CellComplex, PathComplex)):
        if neighborhood_type == "adj":
            ind, A = complex.adjacency_matrix(neighborhood_dim["rank"], index=True)

        else:
            ind, A = complex.coadjacency_matrix(neighborhood_dim["rank"], index=True)
    elif isinstance(complex, (CombinatorialComplex, ColoredHyperGraph)):
        if neighborhood_type == "adj":
            ind, A = complex.adjacency_matrix(
                neighborhood_dim["rank"], neighborhood_dim["via_rank"], index=True
            )
        elif neighborhood_type == "coadj":
            ind, A = complex.coadjacency_matrix(
                neighborhood_dim["rank"], neighborhood_dim["via_rank"], index=True
            )
        else:
            raise TypeError("""Input neighborhood_type  must be 'adj' or 'coadj' .""")
    else:
        raise TypeError(
            """Input Complex can only be a SimplicialComplex, CellComplex, PathComplex ColoredHyperGraph or CombinatorialComplex."""
        )

    return ind, A
