"""Functions for computing neighborhoods of a complex.

This module provides a single public helper, :func:`neighborhood_from_complex`,
for converting local relationships in a TopoNetX complex into sparse matrices.
The returned matrix can represent a same-rank neighborhood, an incidence-based
connection graph, a Hasse graph, or an augmented Hasse graph.

The function is intentionally used as a thin compatibility layer between
TopoNetX complexes and embedding methods in TopoEmbedX. In the original use
case, ``"adj"`` and ``"coadj"`` return the usual same-rank adjacency and
coadjacency matrices. The extended use cases return square sparse graph
matrices whose rows and columns index the cells represented in the graph.
"""

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
RankPair = tuple[int, int]
IncidenceCandidate = tuple[tuple[Any, ...], dict[str, Any]]

_SAME_RANK_NEIGHBORHOODS: set[str] = {"adj", "coadj"}
_CONNECTION_NEIGHBORHOODS: set[str] = {"inc", "incidence", "connection"}
_VALID_NEIGHBORHOODS: set[str] = (
    _SAME_RANK_NEIGHBORHOODS
    | _CONNECTION_NEIGHBORHOODS
    | {"hasse", "augmented_hasse"}
)


def neighborhood_from_complex(
    domain: tnx.Complex,
    neighborhood_type: NeighborhoodType = "adj",
    neighborhood_dim: Mapping[str, Any] | None = None,
) -> tuple[list[Hashable], csr_matrix]:
    """Compute the neighborhood of a complex.

    This function converts a TopoNetX complex into a sparse neighborhood matrix.
    The returned matrix can be used directly as the graph on which a topological
    embedding method operates.

    The function supports three main kinds of neighborhoods.

    First, ``"adj"`` and ``"coadj"`` return same-rank neighborhoods. These are
    the original TopoEmbedX behaviors. For simplicial, cell, and path complexes,
    only ``"rank"`` is used. For combinatorial complexes and colored
    hypergraphs, both ``"rank"`` and ``"via_rank"`` are used.

    Second, ``"inc"``, ``"incidence"``, and ``"connection"`` return square
    connection graphs induced by one or more incidence matrices. A single
    connection graph is specified by ``"rank"`` and ``"to_rank"``. Multiple
    connection graphs can be specified by ``"rank_pairs"`` or ``"pairs"``.
    For example, ``{"rank_pairs": [(0, 1), (1, 2), (0, 2)]}`` builds one
    graph from several cross-rank incidence relations.

    Third, ``"hasse"`` and ``"augmented_hasse"`` return graphs whose nodes are
    cells. The Hasse graph uses incidence relations between selected ranks.
    The augmented Hasse graph starts from the Hasse graph and then adds any
    extra neighborhoods listed under ``"neighborhoods"``.

    Parameters
    ----------
    domain : toponetx.classes.Complex
        The complex to compute the neighborhood for. Supported inputs are
        ``SimplicialComplex``, ``CellComplex``, ``PathComplex``,
        ``CombinatorialComplex``, and ``ColoredHyperGraph``.
    neighborhood_type : {"adj", "coadj", "inc", "incidence", "connection", "hasse", "augmented_hasse"}, default="adj"
        The type of neighborhood to compute.

        ``"adj"``
            Same-rank adjacency. Two rank-``r`` cells are adjacent when they
            share an upper incident cell, according to the TopoNetX adjacency
            convention.

        ``"coadj"``
            Same-rank coadjacency. Two rank-``r`` cells are coadjacent when
            they share a lower incident cell, according to the TopoNetX
            coadjacency convention.

        ``"inc"``, ``"incidence"``, ``"connection"``
            A square graph induced by one or more incidence matrices
            :math:`B_{ij}`. This is useful for cross-rank neighborhoods, such
            as vertices connected to edges or vertices connected directly to
            2-cells in a combinatorial complex.

        ``"hasse"``
            A graph induced by incidence relations between consecutive or
            explicitly selected ranks. This represents the Hasse-style
            cell-incidence structure.

        ``"augmented_hasse"``
            A Hasse graph augmented with additional user-specified
            neighborhoods, such as same-rank adjacency, coadjacency, or
            arbitrary connection graphs.
    neighborhood_dim : mapping, optional
        Parameters specifying the neighborhood. If omitted, the default is
        ``{"rank": 0, "via_rank": -1}``.

        Common keys are:

        ``"rank"`` : int
            Source or target rank used by the requested neighborhood.

        ``"via_rank"`` : int
            Intermediate rank used by TopoNetX adjacency or coadjacency for
            combinatorial complexes and colored hypergraphs. In connection
            graphs, a nonnegative ``"via_rank"`` is treated as an alias for
            ``"to_rank"``.

        ``"to_rank"`` or ``"target_rank"`` : int
            Target rank for a connection graph.

        ``"rank_pairs"`` or ``"pairs"`` : sequence of tuple[int, int]
            Rank pairs used to build several connection graphs at once.

        ``"ranks"`` : sequence of int
            Rank sequence used to build consecutive rank pairs. For example,
            ``{"ranks": [0, 1, 2]}`` is interpreted as rank pairs
            ``(0, 1)`` and ``(1, 2)``.

        ``"neighborhoods"`` : sequence of mapping
            Extra neighborhoods added to an ``"augmented_hasse"`` graph. Each
            entry must contain a ``"type"`` key whose value is one of the
            supported neighborhood types.

        ``"symmetric"`` : bool, default=True
            Whether to add reverse edges when constructing connection, Hasse,
            or augmented Hasse graphs. The default gives an undirected graph
            represented by a symmetric matrix.

        ``"ranked_labels"`` : bool, default=True
            Whether cross-rank graph nodes are labeled as ``(rank, cell)``.
            Keeping this enabled avoids collisions when cells from different
            ranks have similar representations.

    Returns
    -------
    ind : list
        Cell identifiers represented by the rows and columns of ``A``. For
        same-rank neighborhoods, these are the cells of the selected rank. For
        connection, Hasse, and augmented Hasse graphs, these are the cells
        included in the constructed graph. By default, cross-rank labels are
        rank-aware labels of the form ``(rank, cell)``.
    A : scipy.sparse.csr_matrix
        Sparse matrix representing the selected neighborhood graph. The matrix
        is binary and stored in CSR format.

    Raises
    ------
    TypeError
        If ``domain`` is unsupported or ``neighborhood_type`` is invalid.
    ValueError
        If the requested rank parameters are inconsistent.

    Examples
    --------
    Same-rank adjacency on vertices of a cell complex:

    >>> import toponetx as tnx
    >>> from topoembedx.neighborhood import neighborhood_from_complex
    >>> domain = tnx.CellComplex([[0, 1, 2]])
    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="adj",
    ...     neighborhood_dim={"rank": 0},
    ... )
    >>> matrix.shape == (len(ind), len(ind))
    True

    Same-rank coadjacency on edges of a cell complex:

    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="coadj",
    ...     neighborhood_dim={"rank": 1},
    ... )
    >>> matrix.shape == (len(ind), len(ind))
    True

    A connection graph between vertices and edges. The result is square because
    both ranks are represented as nodes in one graph:

    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="connection",
    ...     neighborhood_dim={"rank": 0, "to_rank": 1},
    ... )
    >>> matrix.shape == (len(ind), len(ind))
    True

    The aliases ``"inc"`` and ``"incidence"`` are equivalent to
    ``"connection"``:

    >>> ind_inc, matrix_inc = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="inc",
    ...     neighborhood_dim={"rank": 0, "to_rank": 1},
    ... )
    >>> ind_connection, matrix_connection = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="connection",
    ...     neighborhood_dim={"rank": 0, "to_rank": 1},
    ... )
    >>> ind_inc == ind_connection
    True
    >>> (matrix_inc != matrix_connection).nnz == 0
    True

    A Hasse graph over ranks 0, 1, and 2:

    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="hasse",
    ...     neighborhood_dim={"ranks": [0, 1, 2]},
    ... )
    >>> matrix.shape == (len(ind), len(ind))
    True

    An augmented Hasse graph with extra same-rank neighborhoods:

    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="augmented_hasse",
    ...     neighborhood_dim={
    ...         "ranks": [0, 1, 2],
    ...         "neighborhoods": [
    ...             {"type": "adj", "rank": 0},
    ...             {"type": "coadj", "rank": 1},
    ...         ],
    ...     },
    ... )
    >>> matrix.shape == (len(ind), len(ind))
    True

    A combinatorial complex can use arbitrary cross-rank connection graphs,
    such as a direct :math:`B_{02}` connection between rank-0 and rank-2 cells:

    >>> domain = tnx.CombinatorialComplex()
    >>> domain.add_cell([0], rank=0)
    >>> domain.add_cell([1], rank=0)
    >>> domain.add_cell([2], rank=0)
    >>> domain.add_cell([0, 1], rank=1)
    >>> domain.add_cell([1, 2], rank=1)
    >>> domain.add_cell([0, 2], rank=1)
    >>> domain.add_cell([0, 1, 2], rank=2)
    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="connection",
    ...     neighborhood_dim={"rank": 0, "to_rank": 2},
    ... )
    >>> matrix.shape == (len(ind), len(ind))
    True

    Several connection graphs can be combined at once:

    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="connection",
    ...     neighborhood_dim={"rank_pairs": [(0, 1), (1, 2), (0, 2)]},
    ... )
    >>> matrix.shape == (len(ind), len(ind))
    True

    By default, cross-rank labels include the rank. Set ``"ranked_labels"`` to
    ``False`` to return unranked cell labels:

    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="connection",
    ...     neighborhood_dim={
    ...         "rank": 0,
    ...         "to_rank": 2,
    ...         "ranked_labels": False,
    ...     },
    ... )
    >>> all(not isinstance(cell, tuple) for cell in ind)
    True

    Directed graphs can be requested by disabling symmetrization:

    >>> ind, matrix = neighborhood_from_complex(
    ...     domain,
    ...     neighborhood_type="connection",
    ...     neighborhood_dim={
    ...         "rank": 0,
    ...         "to_rank": 2,
    ...         "symmetric": False,
    ...     },
    ... )
    >>> (matrix != matrix.T).nnz > 0
    True
    """
    neighborhood_dim = _normalize_neighborhood_dim(neighborhood_dim)

    if neighborhood_type not in _VALID_NEIGHBORHOODS:
        raise TypeError(
            "Input neighborhood_type must be one of "
            f"{sorted(_VALID_NEIGHBORHOODS)}, got {neighborhood_type}."
        )

    _validate_complex(domain)

    if neighborhood_type == "adj" or neighborhood_type == "coadj":
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


def _validate_complex(domain: Any) -> None:
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
            "PathComplex ColoredHyperGraph or CombinatorialComplex."
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
        for cell in row_cells:
            add_node(row_rank, cell)
        for cell in col_cells:
            add_node(col_rank, cell)

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

        if local_type == "adj" or local_type == "coadj":
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
    rank_pairs: Sequence[RankPair],
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

        for cell in low_cells:
            add_node(low_rank, cell)
        for cell in high_cells:
            add_node(high_rank, cell)

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
    """Return the incidence matrix from low-rank cells to high-rank cells."""
    if low_rank == high_rank:
        raise ValueError("Incidence rank pairs must contain two distinct ranks.")

    if high_rank < low_rank:
        low_rank, high_rank = high_rank, low_rank

    candidates: list[IncidenceCandidate]
    if isinstance(domain, (tnx.CombinatorialComplex, tnx.ColoredHyperGraph)):
        candidates = [
            ((low_rank, high_rank), {"index": True}),
            ((), {"rank": low_rank, "to_rank": high_rank, "index": True}),
            ((), {"rank": low_rank, "via_rank": high_rank, "index": True}),
            ((), {"from_rank": low_rank, "to_rank": high_rank, "index": True}),
            ((high_rank,), {"index": True}),
        ]
    else:
        if high_rank != low_rank + 1:
            raise ValueError(
                "Non-combinatorial complexes support Hasse incidence only "
                "between consecutive ranks. Use a CombinatorialComplex or "
                "ColoredHyperGraph for arbitrary B_ij connections."
            )

        candidates = [
            ((high_rank,), {"index": True}),
            ((high_rank,), {"signed": False, "index": True}),
            ((), {"rank": high_rank, "index": True}),
            ((), {"rank": high_rank, "signed": False, "index": True}),
        ]

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
) -> list[RankPair]:
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


def _normalize_rank_pairs(rank_pairs: Any) -> list[RankPair]:
    """Normalize a sequence of rank-pair specifications."""
    pairs = [_normalize_rank_pair(pair) for pair in rank_pairs]
    if not pairs:
        raise ValueError("At least one rank pair must be specified.")
    return pairs


def _normalize_rank_pair(rank_pair: Any) -> RankPair:
    """Normalize one rank-pair specification."""
    if not isinstance(rank_pair, Sequence) or len(rank_pair) != 2:
        raise ValueError("Each rank pair must be a two-entry sequence.")

    source_rank, target_rank = int(rank_pair[0]), int(rank_pair[1])
    if source_rank == target_rank:
        raise ValueError("A rank pair must contain two distinct ranks.")

    return source_rank, target_rank


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
        return rank, cell

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
