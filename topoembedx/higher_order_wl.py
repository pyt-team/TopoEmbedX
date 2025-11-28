
import hashlib
from typing import Any, Dict, List, Optional, Union, Literal

import networkx as nx
import numpy as np
import toponetx as tnx
from scipy.sparse import csr_matrix, hstack, vstack



def neighborhood_from_complex(
    domain: tnx.Complex,
    neighborhood_type: Literal["adj", "coadj", "boundary", "coboundary"] = "adj",
    neighborhood_dim: Optional[Dict] = None,
) -> tuple[list, csr_matrix]:
    """
    Compute a neighborhood matrix for a TopoNetX complex.

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
             lower rank: r-1
             upper rank: r
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
        # Simplicial / Cell / Path
        if isinstance(
            domain, tnx.SimplicialComplex | tnx.CellComplex | tnx.PathComplex
        ):
            r = neighborhood_dim["rank"]
            if neighborhood_type == "adj":
                ind, A = domain.adjacency_matrix(r, index=True)
            else:
                ind, A = domain.coadjacency_matrix(r, index=True)

        # Combinatorial / ColoredHyperGraph
        elif isinstance(domain, tnx.CombinatorialComplex | tnx.ColoredHyperGraph):
            r = neighborhood_dim["rank"]
            via = neighborhood_dim.get("via_rank", None)
            if neighborhood_type == "adj":
                ind, A = domain.adjacency_matrix(r, via, index=True)
            else:
                ind, A = domain.coadjacency_matrix(r, via, index=True)

        else:
            raise TypeError(
                "Input Complex can only be a SimplicialComplex, CellComplex, "
                "PathComplex, ColoredHyperGraph or CombinatorialComplex."
            )

        return list(ind), csr_matrix(A).asformat("csr")

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


class HigherOrderWeisfeilerLehmanHashing:
    """
    General multi-neighborhood Weisfeiler–Lehman (WL) refinement.

    Supports:
      (1) Standard graph WL          (domain = networkx.Graph)
      (2) Higher-order WL on complex (one neighborhood)
      (3) SWL-style WL               (multiple neighborhoods A^(r,k))

    Update rule (conceptually):

        c^{t+1}_σ = HASH(
            c^t_σ,
            M^{(1)}_t(σ),
            ...,
            M^{(m)}_t(σ)
        )

    where M^{(i)}_t(σ) is the multiset of neighbor labels under the i-th
    neighborhood matrix A^{(i)}.
    """

    def __init__(self, wl_iterations: int = 3, erase_base_features: bool = False):
        self.wl_iterations = wl_iterations
        self.erase_base_features = erase_base_features

        self.domain = None
        self.nodes: List[Any] = []             # external IDs: vertices / cells
        self.adj_mats: List[csr_matrix] = []   # list of CSR matrices

        self.extracted_features: Dict[Any, List[str]] = {}  # external node -> labels

    # -------------------------------------------------------
    def fit(
        self,
        domain: Union[nx.Graph, "tnx.Complex"],
        neighborhood_types: Union[str, List[str]] = "adj",
        neighborhood_dims: Optional[Union[Dict, List[Optional[Dict]]]] = None,
    ) -> "HigherOrderWeisfeilerLehmanHashing":
        """
        Fit WL on a domain with one or more neighborhoods.

        Parameters
        ----------
        domain : networkx.Graph or TopoNetX complex
        neighborhood_types : str or list[str]
            If domain is a graph:
                - this argument is ignored; plain adjacency is used.
            If domain is a complex:
                - either a single neighborhood type (e.g. "adj")
                - or a list of types (e.g. ["adj","coadj"]).
        neighborhood_dims : dict or list[dict], optional
            Rank specifications (e.g. {"rank": 2, "via_rank": 1}).
        """
        self.domain = domain

        # ---------------------------------------------------
        # CASE 1: Graph domain → standard 1-WL
        # ---------------------------------------------------
        if isinstance(domain, nx.Graph):
            ind = list(domain.nodes())
            self.nodes = list(ind)  # force Python list

            A_arr = nx.to_scipy_sparse_array(domain, nodelist=ind)
            A = csr_matrix(A_arr)   # ensure csr_matrix
            self.adj_mats = [A]

        # ---------------------------------------------------
        # CASE 2: TopoNetX complex → use neighborhood_from_complex
        # ---------------------------------------------------
        else:
            # Normalize types to list
            if isinstance(neighborhood_types, str):
                types_list = [neighborhood_types]
            else:
                types_list = list(neighborhood_types)

            # Normalize dims to list
            if neighborhood_dims is None or isinstance(neighborhood_dims, Dict):
                dims_list = [neighborhood_dims] * len(types_list)
            else:
                dims_list = list(neighborhood_dims)

            if len(types_list) != len(dims_list):
                raise ValueError(
                    "neighborhood_types and neighborhood_dims must have same length."
                )

            ind_lists = []
            mats = []
            for ntype, ndim in zip(types_list, dims_list):
                ind, A_raw = neighborhood_from_complex(
                    domain,
                    neighborhood_type=ntype,
                    neighborhood_dim=ndim,
                )
                # Ensure index is a plain list:
                if not isinstance(ind, list):
                    ind = list(ind)
                # Ensure matrix is csr_matrix:
                A = csr_matrix(A_raw).asformat("csr")

                ind_lists.append(ind)
                mats.append(A)

            # Ensure all index lists are identical (same ordering of cells)
            base = list(ind_lists[0])
            for cur in ind_lists[1:]:
                if list(cur) != base:
                    raise ValueError(
                        "All neighborhoods must use the same index list. "
                        "Reorder externally before calling WL."
                    )

            self.nodes = base[:]     # external IDs (cells or vertices)
            self.adj_mats = mats

        # ---------------------------------------------------
        # Initialize labels on index space {0,...,n-1}
        # ---------------------------------------------------
        self._pos_to_node = {i: node for i, node in enumerate(self.nodes)}

        labels = {i: "0" for i in range(len(self.nodes))}
        self.extracted_features = {node: ["0"] for node in self.nodes}

        # ---------------------------------------------------
        # Run WL iterations
        # ---------------------------------------------------
        for _ in range(self.wl_iterations):
            labels = self._wl_step(labels)

        # Optionally remove base label
        if self.erase_base_features:
            for node in self.extracted_features:
                if self.extracted_features[node]:
                    del self.extracted_features[node][0]

        return self

    # -------------------------------------------------------
    def _wl_step(self, labels: Dict[int, str]) -> Dict[int, str]:
        """One WL refinement step in index space."""
        new_labels: Dict[int, str] = {}
        n = len(self.nodes)

        for pos in range(n):
            parts = [labels[pos]]  # own current label

            # Aggregate multiset of neighbor labels for each neighborhood
            for A in self.adj_mats:
                row = A.getrow(pos)
                neigh_idx = row.indices.tolist()
                neigh_labels = sorted(labels[j] for j in neigh_idx)
                parts.append("_".join(neigh_labels))

            concat = "|".join(parts)
            hashed = hashlib.md5(concat.encode()).hexdigest()
            new_labels[pos] = hashed

            ext_node = self._pos_to_node[pos]
            self.extracted_features[ext_node].append(hashed)

        return new_labels

    # -------------------------------------------------------
    def get_cell_features(self) -> Dict[Any, List[str]]:
        """Return dict: external cell/node → list of WL labels over iterations."""
        return self.extracted_features

    def get_domain_features(self) -> List[str]:
        """Return flattened multiset of labels across all cells and iterations."""
        return [f for feats in self.extracted_features.values() for f in feats]
