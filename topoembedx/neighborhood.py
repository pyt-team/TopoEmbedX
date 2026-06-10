"""Functions for computing neighborhoods of a complex."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import toponetx as tnx
from scipy.sparse import coo_matrix, csr_matrix

NeighborhoodType = Literal[
    "adj",
    "coadj",
    "inc",
    "incidence",
    "connection",
    "hasse",
    "augmented_hasse",
]

_SAME_RANK_NEIGHBORHOODS = {"adj", "coadj"}
_CONNECTION_NEIGHBORHOODS = {"inc", "incidence", "connection"}
_VALID_NEIGHBORHOODS = (
    _SAME_RANK_NEIGHBORHOODS
    | _CONNECTION_NEIGHBORHOODS
    | {"hasse", "augmented_hasse"}
)


def neighborhood_from_complex(
    domain: tnx.Complex,
    neighborhood_type: NeighborhoodType = "adj",
    neighborhood_dim: Mapping[str, Any] | None = None,
) -> tuple[list[Hashable], csr_matrix]:
    """Compute a neighborhood matrix from a complex.

    This function returns the indices and sparse matrix for the neighborhood
    specified by ``neighborhood_type`` and ``neighborhood_dim``. The original
    ``"adj"`` and ``"coadj"`` cases are preserved. Additional cases construct
    connection, Hasse, or augmented Hasse graphs by assembling sparse
    incidence and neighborhood matrices.

    Parameters
    ----------
    domain : toponetx.classes.Complex
        The complex to compute the neighborhood for.
    neighborhood_type : {"adj", "coadj", "inc", "incidence", "connection", "hasse", "augmented_hasse"}, default="adj"
        The type of neighborhood to compute.

        - ``"adj"`` returns a same-rank adjacency matrix.
        - ``"coadj"`` returns a same-rank coadjacency matrix.
        - ``"inc"``, ``"incidence"``, or ``"connection"`` returns a square
          connection graph induced by one or more incidence matrices
          :math:`B_{ij}`.
        - ``"hasse"`` returns the graph induced by cover/incidence relations
          between selected ranks.
        - ``"augmented_hasse"`` returns the union of the Hasse graph and any
          extra neighborhoods listed in ``neighborhood_dim["neighborhoods"]``.
    neighborhood_dim : mapping, optional
        Integer parameters specifying the neighborhood. For the original
        same-rank cases, use ``{"rank": r, "via_rank": s}``, where
        ``"via_rank"`` is used only for combinatorial complexes and colored
        hypergraphs.

        For connection, Hasse, or augmented Hasse graphs, the following keys
        are supported:

        - ``"rank"`` and ``"to_rank"``: construct one :math:`B_{ij}` graph.
        - ``"rank_pairs"``: construct multiple :math:`B_{ij}` graphs, for
          example ``[(0, 1), (1, 2), (0, 2)]``.
        - ``"ranks"``: construct consecutive pairs from the listed ranks.
        - ``"neighborhoods"``: for ``"augmented_hasse"``, add extra
          neighborhoods. Each entry is a mapping with a ``"type"`` key and the
          corresponding rank parameters, for example
          ``{"type": "coadj", "rank": 1}``.

    Returns
    -------
    ind : list
        Cell identifiers represented by the rows and columns of ``A``. For
        cross-rank graphs, entries are rank-labeled pairs ``(rank, cell)`` when
        ``neighborhood_dim["ranked_labels"]`` is true.
    A : scipy.sparse.csr_matrix
        Sparse matrix representing the selected neighborhood graph.

    Raises
    ------
    TypeError
        If ``domain`` is unsupported or ``neighborhood_type`` is invalid.
    ValueError
        If the requested rank parameters are inconsistent.
    """
    neighborhood_dim = _normalize_neighborhood_dim(neighborhood_dim)

    if neighborhood_type not in _VALID_NEIGHBORHOODS:
        raise TypeError(
            "Input neighborhood_type must be one of "
            f"{sorted(_VALID_NEIGHBORHOODS)}, got {neighborhood_type}."
        )

    _validate_complex(domain)

    if neighborhood_type in _SAME_RANK_NEIGHBORHOODS:
        return _same_rank_neighborhood(domain, neighborhood_type, neighborhood_dim)

    if neighborhood_type in _CONNECTION_NEIGHBORHOODS:
        rank_pairs = _rank_pairs_from_neighborhood_dim(
            domain,
            neighborhood_dim,
            default_all_consecutive=False,
        )
        return _connection_graph_from_rank_pairs(
            domain,
            rank_pairs,
            neighborhood_dim,
        )

    if neighborhood_type == "hasse":
        rank_pairs = _rank_pairs_from_neighborhood_dim(
            domain,
            neighborhood_dim,
            default_all_consecutive=True,
        )
        return _connection_graph_from_rank_pairs(
            domain,
            rank_pairs,
            neighborhood_dim,
        )

    return _augmented_hasse_graph(domain, neighborhood_dim)


def _normalize_neighborhood_dim(
    neighborhood_dim: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a mutable neighborhood-parameter dictionary."""
    if neighborhood_dim is None:
        return {"rank": 0, "via_rank": -1}

    return dict(neighborhood_dim)


def _validate_complex(domain: tnx.Complex) -> None:
    """Validate the input complex type."""
    valid_complexes = (
        tnx.SimplicialComplex,
        tnx.CellComplex,
        tnx.PathComplex,
        tnx.CombinatorialComplex,
        tnx.ColoredHyperGraph,
    )

    if not isinstance(domain, valid_complexes):
        raise TypeError(
            "Input Complex can only be a SimplicialComplex, CellComplex, "
            "PathComplex, ColoredHyperGraph, or CombinatorialComplex."
        )


def _same_rank_neighborhood(
    domain: tnx.Complex,
    neighborhood_type: Literal["adj", "coadj"],
    neighborhood_dim: Mapping[str, Any],
) -> tuple[list[Hashable], csr_matrix]:
    """Compute a same-rank adjacency or coadjacency neighborhood."""
    rank = int(neighborhood_dim["rank"])

    if isinstance(domain, (tnx.SimplicialComplex, tnx.CellComplex, tnx.PathComplex)):
        if neighborhood_type == "adj":
            ind, matrix = domain.adjacency_matrix(rank, index=True)
        else:
            ind, matrix = domain.coadjacency_matrix(rank, index=True)
    elif isinstance(domain, (tnx.CombinatorialComplex, tnx.ColoredHyperGraph)):
        via_rank = int(neighborhood_dim["via_rank"])
        if neighborhood_type == "adj":
            ind, matrix = domain.adjacency_matrix(rank, via_rank, index=True)
        else:
            ind, matrix = domain.coadjacency_matrix(rank, via_rank, index=True)
    else:
        raise TypeError("Unsupported complex type.")

    return _ordered_cells(ind), _binary_csr(matrix)


def _augmented_hasse_graph(
    domain: tnx.Complex,
    neighborhood_dim: Mapping[str, Any],
) -> tuple[list[Hashable], csr_matrix]:
    """Construct an augmented Hasse graph from Hasse and extra neighborhoods."""
    rank_pairs = _rank_pairs_from_neighborhood_dim(
        domain,
        neighborhood_dim,
        default_all_consecutive=True,
    )
    ranked_labels = bool(neighborhood_dim.get("ranked_labels", True))
    symmetric = bool(neighborhood_dim.get("symmetric", True))

    nodes: list[Hashable] = []
    node_to_position: dict[Hashable, int] = {}
    rows: list[int] = []
    cols: list[int] = []

    def add_node(rank: int, cell: Hashable) -> int:
        node = _node_label(rank, cell, ranked_labels)
        if node not in node_to_position:
            node_to_position[node] = len(nodes)
            nodes.append(node)
        return node_to_position[node]

    def add_edges_from_block(
        row_rank: int,
        row_cells: Sequence[Hashable],
        col_rank: int,
        col_cells: Sequence[Hashable],
        matrix: csr_matrix,
    ) -> None:
        block = _binary_csr(matrix).tocoo()
        for row, col in zip(block.row, block.col, strict=True):
            source = add_node(row_rank, row_cells[int(row)])
            target = add_node(col_rank, col_cells[int(col)])
            rows.append(source)
            cols.append(target)
            if symmetric:
                rows.append(target)
                cols.append(source)

    for low_rank, high_rank in rank_pairs:
        low_cells, high_cells, matrix = _incidence_between_ranks(
            domain,
            low_rank,
            high_rank,
        )
        add_edges_from_block(low_rank, low_cells, high_rank, high_cells, matrix)

    for neighborhood in neighborhood_dim.get("neighborhoods", []):
        if not isinstance(neighborhood, Mapping):
            raise TypeError("Each augmented neighborhood must be a mapping.")

        local_dim = dict(neighborhood)
        local_type = local_dim.pop("type", None)
        if local_type not in _VALID_NEIGHBORHOODS:
            raise TypeError(
                "Each augmented neighborhood must define a valid `type`, got "
                f"{local_type}."
            )

        if local_type in _SAME_RANK_NEIGHBORHOODS:
            rank = int(local_dim["rank"])
            local_ind, local_matrix = _same_rank_neighborhood(
                domain,
                local_type,
                local_dim,
            )
            add_edges_from_block(rank, local_ind, rank, local_ind, local_matrix)
            continue

        local_pairs = _rank_pairs_from_neighborhood_dim(
            domain,
            local_dim,
            default_all_consecutive=local_type in {"hasse", "augmented_hasse"},
        )
        for low_rank, high_rank in local_pairs:
            low_cells, high_cells, matrix = _incidence_between_ranks(
                domain,
                low_rank,
                high_rank,
            )
            add_edges_from_block(low_rank, low_cells, high_rank, high_cells, matrix)

    shape = (len(nodes), len(nodes))
    data = np.ones(len(rows), dtype=np.int8)
    matrix = coo_matrix((data, (rows, cols)), shape=shape, dtype=np.int8).tocsr()
    return nodes, _binary_csr(matrix)


def _connection_graph_from_rank_pairs(
    domain: tnx.Complex,
    rank_pairs: Sequence[tuple[int, int]],
    neighborhood_dim: Mapping[str, Any],
) -> tuple[list[Hashable], csr_matrix]:
    """Construct a square graph from one or more incidence matrices."""
    ranked_labels = bool(neighborhood_dim.get("ranked_labels", True))
    symmetric = bool(neighborhood_dim.get("symmetric", True))

    nodes: list[Hashable] = []
    node_to_position: dict[Hashable, int] = {}
    rows: list[int] = []
    cols: list[int] = []

    def add_node(rank: int, cell: Hashable) -> int:
        node = _node_label(rank, cell, ranked_labels)
        if node not in node_to_position:
            node_to_position[node] = len(nodes)
            nodes.append(node)
        return node_to_position[node]

    for low_rank, high_rank in rank_pairs:
        low_cells, high_cells, matrix = _incidence_between_ranks(
            domain,
            low_rank,
            high_rank,
        )
        block = _binary_csr(matrix).tocoo()

        for row, col in zip(block.row, block.col, strict=True):
            source = add_node(low_rank, low_cells[int(row)])
            target = add_node(high_rank, high_cells[int(col)])
            rows.append(source)
            cols.append(target)
            if symmetric:
                rows.append(target)
                cols.append(source)

    shape = (len(nodes), len(nodes))
    data = np.ones(len(rows), dtype=np.int8)
    matrix = coo_matrix((data, (rows, cols)), shape=shape, dtype=np.int8).tocsr()
    return nodes, _binary_csr(matrix)


def _incidence_between_ranks(
    domain: tnx.Complex,
    low_rank: int,
    high_rank: int,
) -> tuple[list[Hashable], list[Hashable], csr_matrix]:
    """Return the incidence matrix from ``low_rank`` cells to ``high_rank`` cells."""
    if low_rank == high_rank:
        raise ValueError("Incidence rank pairs must contain two distinct ranks.")

    if high_rank < low_rank:
        low_rank, high_rank = high_rank, low_rank

    if isinstance(domain, (tnx.CombinatorialComplex, tnx.ColoredHyperGraph)):
        candidates = (
            ((low_rank, high_rank), {"index": True}),
            ((), {"rank": low_rank, "to_rank": high_rank, "index": True}),
            ((), {"rank": low_rank, "via_rank": high_rank, "index": True}),
            ((), {"from_rank": low_rank, "to_rank": high_rank, "index": True}),
            ((high_rank,), {"index": True}),
        )
    else:
        if high_rank != low_rank + 1:
            raise ValueError(
                "Non-combinatorial complexes support Hasse incidence only "
                "between consecutive ranks. Use a CombinatorialComplex or "
                "ColoredHyperGraph for arbitrary B_ij connections."
            )

        candidates = (
            ((high_rank,), {"index": True}),
            ((high_rank,), {"signed": False, "index": True}),
            ((), {"rank": high_rank, "index": True}),
            ((), {"rank": high_rank, "signed": False, "index": True}),
        )

    method = domain.incidence_matrix
    last_error: TypeError | None = None

    for args, kwargs in candidates:
        try:
            result = method(*args, **kwargs)
        except TypeError as error:
            last_error = error
            continue

        row_cells, col_cells, matrix = _unpack_incidence_result(result)
        row_cells, col_cells, matrix = _orient_incidence_result(
            domain,
            low_rank,
            high_rank,
            row_cells,
            col_cells,
            matrix,
        )
        return row_cells, col_cells, _binary_csr(matrix)

    raise TypeError(
        "Unable to compute an incidence matrix with the available TopoNetX API."
    ) from last_error


def _unpack_incidence_result(
    result: Any,
) -> tuple[list[Hashable], list[Hashable], csr_matrix]:
    """Unpack an incidence-matrix result returned by TopoNetX."""
    if not isinstance(result, tuple):
        raise TypeError("Expected TopoNetX to return a tuple.")

    if len(result) == 3:
        row_index, col_index, matrix = result
        return _ordered_cells(row_index), _ordered_cells(col_index), _binary_csr(matrix)

    if len(result) == 2:
        index, matrix = result
        if isinstance(index, tuple) and len(index) == 2:
            row_index, col_index = index
            return (
                _ordered_cells(row_index),
                _ordered_cells(col_index),
                _binary_csr(matrix),
            )

    raise TypeError(
        "Expected incidence_matrix(..., index=True) to return "
        "(row_index, col_index, matrix) or ((row_index, col_index), matrix)."
    )


def _orient_incidence_result(
    domain: tnx.Complex,
    low_rank: int,
    high_rank: int,
    row_cells: list[Hashable],
    col_cells: list[Hashable],
    matrix: csr_matrix,
) -> tuple[list[Hashable], list[Hashable], csr_matrix]:
    """Orient incidence rows as lower-rank cells and columns as higher-rank cells."""
    low_cells = _cells_of_rank(domain, low_rank)
    high_cells = _cells_of_rank(domain, high_rank)

    if not low_cells or not high_cells:
        return row_cells, col_cells, matrix

    row_set = {_hashable_cell(cell) for cell in row_cells}
    col_set = {_hashable_cell(cell) for cell in col_cells}
    low_set = {_hashable_cell(cell) for cell in low_cells}
    high_set = {_hashable_cell(cell) for cell in high_cells}

    if row_set == low_set and col_set == high_set:
        return row_cells, col_cells, matrix

    if row_set == high_set and col_set == low_set:
        return col_cells, row_cells, matrix.transpose().tocsr()

    return row_cells, col_cells, matrix


def _rank_pairs_from_neighborhood_dim(
    domain: tnx.Complex,
    neighborhood_dim: Mapping[str, Any],
    *,
    default_all_consecutive: bool,
) -> list[tuple[int, int]]:
    """Extract rank pairs from neighborhood parameters."""
    if "rank_pairs" in neighborhood_dim:
        return _normalize_rank_pairs(neighborhood_dim["rank_pairs"])

    if "pairs" in neighborhood_dim:
        return _normalize_rank_pairs(neighborhood_dim["pairs"])

    if "ranks" in neighborhood_dim:
        ranks = [int(rank) for rank in neighborhood_dim["ranks"]]
        if len(ranks) < 2:
            raise ValueError("`ranks` must contain at least two ranks.")
        return [(ranks[index], ranks[index + 1]) for index in range(len(ranks) - 1)]

    if "rank" in neighborhood_dim:
        rank = int(neighborhood_dim["rank"])
        target_rank = neighborhood_dim.get("to_rank")
        if target_rank is None:
            target_rank = neighborhood_dim.get("target_rank")
        if target_rank is None and int(neighborhood_dim.get("via_rank", -1)) >= 0:
            target_rank = neighborhood_dim["via_rank"]
        if target_rank is not None:
            return [_normalize_rank_pair((rank, int(target_rank)))]
        if not default_all_consecutive:
            return [_normalize_rank_pair((rank, rank + 1))]

    max_rank = _domain_dimension(domain)
    if default_all_consecutive and max_rank is not None and max_rank > 0:
        return [(rank, rank + 1) for rank in range(max_rank)]

    rank = int(neighborhood_dim.get("rank", 0))
    return [_normalize_rank_pair((rank, rank + 1))]


def _normalize_rank_pairs(rank_pairs: Any) -> list[tuple[int, int]]:
    """Normalize a sequence of rank-pair specifications."""
    pairs = [_normalize_rank_pair(pair) for pair in rank_pairs]
    if not pairs:
        raise ValueError("At least one rank pair must be specified.")
    return pairs


def _normalize_rank_pair(rank_pair: Any) -> tuple[int, int]:
    """Normalize one rank-pair specification."""
    if not isinstance(rank_pair, Sequence) or len(rank_pair) != 2:
        raise ValueError("Each rank pair must be a two-entry sequence.")

    source_rank, target_rank = int(rank_pair[0]), int(rank_pair[1])
    if source_rank == target_rank:
        raise ValueError("A rank pair must contain two distinct ranks.")

    return (source_rank, target_rank)


def _domain_dimension(domain: tnx.Complex) -> int | None:
    """Return the dimension of a complex when exposed by TopoNetX."""
    for attribute in ("dim", "dimension"):
        value = getattr(domain, attribute, None)
        if value is None:
            continue
        if callable(value):
            try:
                return int(value())
            except TypeError:
                continue
        return int(value)

    return None


def _cells_of_rank(domain: tnx.Complex, rank: int) -> list[Hashable]:
    """Return cells of a fixed rank when exposed by TopoNetX."""
    skeleton = getattr(domain, "skeleton", None)
    if skeleton is None:
        return []

    try:
        cells = skeleton(rank)
    except TypeError:
        cells = skeleton(rank=rank)

    return [_hashable_cell(cell) for cell in cells]


def _ordered_cells(index: Any) -> list[Hashable]:
    """Convert a TopoNetX index object into an ordered cell list."""
    if isinstance(index, Mapping):
        return [
            _hashable_cell(cell)
            for cell, _ in sorted(index.items(), key=lambda item: item[1])
        ]

    return [_hashable_cell(cell) for cell in index]


def _node_label(rank: int, cell: Hashable, ranked_labels: bool) -> Hashable:
    """Return the node label used in cross-rank graphs."""
    cell = _hashable_cell(cell)
    if ranked_labels:
        return (rank, cell)

    return cell


def _hashable_cell(cell: Any) -> Hashable:
    """Return a hashable representation of a cell identifier."""
    try:
        hash(cell)
    except TypeError:
        if isinstance(cell, set):
            return frozenset(cell)
        if isinstance(cell, Sequence) and not isinstance(cell, (str, bytes)):
            return tuple(cell)
        return repr(cell)

    return cell


def _binary_csr(matrix: Any) -> csr_matrix:
    """Return a binary CSR matrix with explicit zeros removed."""
    matrix = csr_matrix(matrix)
    matrix.data = np.ones_like(matrix.data, dtype=np.int8)
    matrix.eliminate_zeros()
    return matrix
