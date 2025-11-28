"""Functions for computing neighborhoods of a complex."""

from typing import Literal

import toponetx as tnx
from scipy.sparse import csr_matrix, hstack, vstack


def neighborhood_from_complex(
    domain: tnx.Complex,
    neighborhood_type: Literal["adj", "coadj", "boundary", "coboundary"] = "adj",
    neighborhood_dim=None,
) -> tuple[list, csr_matrix]:
    """Compute a neighborhood matrix for a TopoNetX complex.

    This function returns the indices and matrix for the neighborhood specified by
    ``neighborhood_type`` and ``neighborhood_dim`` for the input complex ``domain``.

    Supported neighborhood types
    ----------------------------
    1. ``"adj"`` (adjacency on a single rank)
       - Nodes are cells of a fixed rank r.
       - Two r-cells are adjacent if they share suitable (co)faces, as defined
         by TopoNetX's `adjacency_matrix` implementation.

    2. ``"coadj"`` (coadjacency on a single rank)
       - Similar to `"adj"`, but using the complex's `coadjacency_matrix`.

    3. ``"boundary"`` / ``"coboundary"`` (Hasse graph from incidence)
       - Here we use the complex's `incidence_matrix` to build an undirected
         **Hasse graph** between two consecutive ranks:
           • lower rank: r-1
           • upper rank: r
       - Nodes are the union of all (r-1)-cells and all r-cells.
       - Edges connect an (r-1)-cell to an r-cell whenever the incidence is nonzero.

       For the purposes of an undirected neighborhood, `"boundary"` and
       `"coboundary"` return the **same** Hasse adjacency; they are conceptually
       different (downward vs upward), but the graph is the same.

    Parameters
    ----------
    domain : toponetx.classes.Complex
        The complex to compute the neighborhood for.
        Must be one of:
        - SimplicialComplex
        - CellComplex
        - PathComplex
        - CombinatorialComplex
        - ColoredHyperGraph
    neighborhood_type : {"adj", "coadj", "boundary", "coboundary"}, default="adj"
        The type of neighborhood to compute.
    neighborhood_dim : dict, optional
        Integer parameters specifying which rank(s) to use.

        For "adj"/"coadj":
        ------------------
        - For Simplicial/Cell/Path:
              neighborhood_dim["rank"]
          selects the rank r whose cells will be the nodes.

        - For Combinatorial/ColoredHyperGraph:
              neighborhood_dim["rank"], neighborhood_dim["via_rank"]
          specify both the rank r of the nodes and the intermediate rank
          via which adjacency/coadjacency is computed.

        For "boundary"/"coboundary":
        ----------------------------
        We use the **incidence** between rank r and rank r-1 via
        `domain.incidence_matrix`:

        - For Simplicial/Cell/Path:
              domain.incidence_matrix(rank=r, index=True)
          is assumed to return:
              ind_low, ind_high, B
          where
              * ind_low  : labels of (r-1)-cells,
              * ind_high : labels of r-cells,
              * B        : incidence matrix (shape n_low X n_high).

        - For Combinatorial/ColoredHyperGraph:
              domain.incidence_matrix(rank=r, via_rank=r-1, index=True)
          is assumed to return:
              ind_low, ind_high, B
          with the same semantics.

        If ``neighborhood_dim`` is None, we default to:
            neighborhood_dim = {"rank": 0, "via_rank": -1}

    Returns
    -------
    ind : list
        A list of the indices for the nodes in the neighborhood graph.
        - For "adj"/"coadj": indices of rank-r cells.
        - For "boundary"/"coboundary": indices of rank-(r-1) cells followed
          by indices of rank-r cells (Hasse graph nodes).
    A : scipy.sparse.csr_matrix
        The matrix representing the neighborhood.
        - For "adj"/"coadj": square (n_r X n_r) adjacency/coadjacency matrix.
        - For "boundary"/"coboundary":
          square ((n_{r-1} + n_r) X (n_{r-1} + n_r)) adjacency matrix of the
          bipartite Hasse graph induced by the incidence matrix.

    Raises
    ------
    TypeError
        If `domain` is not a supported complex type.
    TypeError
        If `neighborhood_type` is invalid.
    """
    # Default neighborhood dimensions
    if neighborhood_dim is None:
        neighborhood_dim = {"rank": 0, "via_rank": -1}

    if neighborhood_type not in ["adj", "coadj", "boundary", "coboundary"]:
        raise TypeError(
            "Input neighborhood_type must be one of "
            "'adj', 'coadj', 'boundary', or 'coboundary', "
            f"got {neighborhood_type}."
        )

    # ------------------------------------------------------------
    # Case 1: adjacency / coadjacency on a fixed rank
    # ------------------------------------------------------------
    if neighborhood_type in ["adj", "coadj"]:
        if isinstance(
            domain, tnx.SimplicialComplex | tnx.CellComplex | tnx.PathComplex
        ):
            if neighborhood_type == "adj":
                ind, A = domain.adjacency_matrix(neighborhood_dim["rank"], index=True)
            else:
                ind, A = domain.coadjacency_matrix(neighborhood_dim["rank"], index=True)

        elif isinstance(domain, tnx.CombinatorialComplex | tnx.ColoredHyperGraph):
            if neighborhood_type == "adj":
                ind, A = domain.adjacency_matrix(
                    neighborhood_dim["rank"],
                    neighborhood_dim["via_rank"],
                    index=True,
                )
            else:
                ind, A = domain.coadjacency_matrix(
                    neighborhood_dim["rank"],
                    neighborhood_dim["via_rank"],
                    index=True,
                )
        else:
            raise TypeError(
                "Input Complex can only be a SimplicialComplex, CellComplex, "
                "PathComplex, ColoredHyperGraph or CombinatorialComplex."
            )

        return ind, A.asformat("csr")

    # ------------------------------------------------------------
    # Case 2: boundary / coboundary → Hasse graph from incidence_matrix
    # ------------------------------------------------------------
    if not hasattr(domain, "incidence_matrix"):
        raise TypeError(
            "The given complex does not provide an 'incidence_matrix' method, "
            "so 'boundary'/'coboundary' neighborhoods are not supported."
        )

    r = neighborhood_dim["rank"]

    # Two cases: (SC / Cell / Path) vs (CC / ColoredHyperGraph)
    if isinstance(domain, tnx.SimplicialComplex | tnx.CellComplex | tnx.PathComplex):
        # Expected signature:
        #   ind_low, ind_high, B = domain.incidence_matrix(rank=r, index=True)
        ind_low, ind_high, B = domain.incidence_matrix(r, index=True)  # type: ignore[arg-type]
    elif isinstance(domain, tnx.CombinatorialComplex | tnx.ColoredHyperGraph):
        # Expected signature:
        #   ind_low, ind_high, B = domain.incidence_matrix(rank=r, via_rank=r-1, index=True)
        via = neighborhood_dim.get("via_rank", r - 1)
        ind_low, ind_high, B = domain.incidence_matrix(  # type: ignore[arg-type]
            rank=r, via_rank=via, index=True
        )
    else:
        raise TypeError(
            "Input Complex can only be a SimplicialComplex, CellComplex, "
            "PathComplex, ColoredHyperGraph or CombinatorialComplex."
        )

    # Make sure B is CSR and unsigned
    B = abs(B).asformat("csr")
    n_low, n_high = B.shape

    # Build bipartite Hasse adjacency:
    #   [ 0   B ]
    #   [ B^T 0 ]
    zero_low = csr_matrix((n_low, n_low))
    zero_high = csr_matrix((n_high, n_high))

    upper = hstack([zero_low, B])
    lower = hstack([B.transpose(), zero_high])
    A_hasse = vstack([upper, lower]).asformat("csr")

    # Node indices = (r-1)-cells followed by r-cells
    ind = list(ind_low) + list(ind_high)

    return ind, A_hasse
