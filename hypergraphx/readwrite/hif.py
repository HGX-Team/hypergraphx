import copy
import json
import sys
from itertools import count
from typing import Literal, TypeAlias, cast

from hypergraphx import DirectedHypergraph, Hypergraph

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:  # python 3.10 doesn't have NotRequired which is very useful for the HIF format
    from typing_extensions import NotRequired, TypedDict

__all__ = [
    "HIFJson",
    "HIFEdgeRecord",
    "HIFIncidenceRecord",
    "HIFNodeRecord",
    "JSONValue",
    "from_hif_dict",
    "read_hif",
    "to_hif_dict",
    "write_hif",
]

HIF_ID: TypeAlias = str | int
Weight: TypeAlias = int | float
JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
Metadata: TypeAlias = dict[str, JSONValue]
NetworkType: TypeAlias = Literal["asc", "directed", "undirected"]
Direction: TypeAlias = Literal["head", "tail"]
UndirectedEdge: TypeAlias = tuple[HIF_ID, ...]
DirectedEdge: TypeAlias = tuple[tuple[HIF_ID, ...], tuple[HIF_ID, ...]]
EdgeKey: TypeAlias = UndirectedEdge | DirectedEdge
UndirectedEdgeMembers: TypeAlias = list[HIF_ID]
DirectedEdgeMembers: TypeAlias = tuple[list[HIF_ID], list[HIF_ID]]


class HIFNodeRecord(TypedDict):
    node: HIF_ID
    weight: NotRequired[Weight]
    attrs: NotRequired[Metadata]


class HIFEdgeRecord(TypedDict):
    edge: HIF_ID
    weight: NotRequired[Weight]
    attrs: NotRequired[Metadata]


class HIFIncidenceRecord(TypedDict):
    edge: HIF_ID
    node: HIF_ID
    weight: NotRequired[Weight]
    direction: NotRequired[Direction]
    attrs: NotRequired[Metadata]


# network-type is not a valid python identifier, so we need to create
# the typedict manually.
HIFJson = TypedDict(
    "HIFJson",
    {
        "network-type": NotRequired[NetworkType],
        "metadata": NotRequired[Metadata],
        "incidences": list[HIFIncidenceRecord],
        "nodes": NotRequired[list[HIFNodeRecord]],
        "edges": NotRequired[list[HIFEdgeRecord]],
    },
)


def _get_record_metadata(
    record: HIFNodeRecord | HIFEdgeRecord | HIFIncidenceRecord,
    include_weight: bool = False,
) -> Metadata:
    """Copy the attributes from a HIF record."""
    attrs = copy.deepcopy(record.get("attrs", {}))
    if not isinstance(attrs, dict):
        raise TypeError("HIF record 'attrs' must be a dictionary.")
    if "weight" in record and type(record["weight"]) not in (int, float):
        raise TypeError("HIF weights must be integers or floats.")
    if include_weight and "weight" in record:
        attrs["weight"] = record["weight"]
    return attrs


def _add_metadata_in_record(
    record: HIFNodeRecord | HIFEdgeRecord | HIFIncidenceRecord,
    metadata: Metadata,
    include_weight: bool = False,
) -> None:
    attrs = copy.deepcopy(metadata)
    if include_weight and "weight" in attrs:
        weight = attrs.pop("weight")
        if type(weight) not in (int, float):
            raise TypeError("HIF weights must be integers or floats.")
        record["weight"] = cast(Weight, weight)
    if attrs:
        record["attrs"] = attrs


def _get_hif_edge_ids(
    internal_edge_ids: dict[EdgeKey, int], reserved_ids: set[HIF_ID]
) -> dict[EdgeKey, HIF_ID]:
    """Keep internal IDs when possible and replace IDs reserved by empty edges."""
    used_ids = reserved_ids.copy()
    available_ids = (candidate for candidate in count() if candidate not in used_ids)
    hif_edge_ids: dict[EdgeKey, HIF_ID] = {}

    for edge_key, internal_id in internal_edge_ids.items():
        hif_edge_id = (
            internal_id if internal_id not in used_ids else next(available_ids)
        )
        hif_edge_ids[edge_key] = hif_edge_id
        used_ids.add(hif_edge_id)

    return hif_edge_ids


def _get_hif_incidences(
    h: Hypergraph | DirectedHypergraph, edge_key: EdgeKey
) -> list[tuple[HIF_ID, Direction | None]]:
    if isinstance(h, DirectedHypergraph):
        source, target = cast(DirectedEdge, edge_key)
        incidences: list[tuple[HIF_ID, Direction | None]] = [
            (node, "tail") for node in source
        ]
        incidences.extend((node, "head") for node in target)
        return incidences

    members = cast(UndirectedEdge, edge_key)
    return [(node, None) for node in members]


def _get_edge_weight(h: Hypergraph | DirectedHypergraph, edge_key: EdgeKey) -> Weight:
    if isinstance(h, DirectedHypergraph):
        return h.get_weight(cast(DirectedEdge, edge_key))
    return h.get_weight(cast(UndirectedEdge, edge_key))


def from_hif_dict(data: HIFJson) -> Hypergraph | DirectedHypergraph:
    """
    Create a hypergraph from a dictionary following the HIF standard.

    Parameters
    ----------
    data : HIFJson
        A HIF dictionary containing an ``incidences`` list and, optionally,
        network type, metadata, nodes, and edges.

    Returns
    -------
    Hypergraph or DirectedHypergraph
        The hypergraph represented by ``data``. If ``network-type`` is omitted,
        an undirected hypergraph is returned.

    Raises
    ------
    TypeError
        If any field has an invalid type according to the HIF schema.
    NotImplementedError
        If ``network-type`` is ``"asc"``.

    Notes
    -----
    HypergraphX does not support parallel edges. Parallel HIF edges are
    rejected rather than merged. Input metadata is deeply copied.
    """
    if not isinstance(data, dict):
        raise TypeError("HIF data must be provided as a dictionary.")
    network_type: NetworkType = data.get("network-type", "undirected")
    edge_records: list[HIFEdgeRecord] = data.get("edges", [])
    is_weighted = any("weight" in record for record in edge_records)
    match network_type:
        case "undirected":
            h = Hypergraph(weighted=is_weighted, duplicate_policy="error")
        case "directed":
            h = DirectedHypergraph(weighted=is_weighted, duplicate_policy="error")
        case "asc":
            raise NotImplementedError(
                "HypergraphX does not support abstract simplicial complexes."
            )
        case _:
            raise ValueError(f"Unknown hypergraph type: {network_type}")

    metadata = copy.deepcopy(data.get("metadata", {}))
    if not isinstance(metadata, dict):
        raise TypeError("HIF 'metadata' must be a dictionary.")
    h.set_hypergraph_metadata(metadata)

    for record in data.get("nodes", []):
        node = record["node"]
        h.add_node(node)
        h.set_node_metadata(node, _get_record_metadata(record, include_weight=True))

    undirected_members_by_hif_edge_id: dict[HIF_ID, UndirectedEdgeMembers] = {}
    directed_members_by_hif_edge_id: dict[HIF_ID, DirectedEdgeMembers] = {}
    for incidence in data["incidences"]:
        hif_edge_id = incidence["edge"]
        node = incidence["node"]
        match h, incidence.get("direction"):
            case DirectedHypergraph(), "tail":
                tail, _ = directed_members_by_hif_edge_id.setdefault(
                    hif_edge_id, ([], [])
                )
                tail.append(node)
            case DirectedHypergraph(), "head":
                _, head = directed_members_by_hif_edge_id.setdefault(
                    hif_edge_id, ([], [])
                )
                head.append(node)
            case DirectedHypergraph(), _:
                raise ValueError(
                    "Directed HIF incidences require direction 'head' or 'tail'."
                )
            case Hypergraph(), _:
                undirected_members_by_hif_edge_id.setdefault(hif_edge_id, []).append(
                    node
                )

    added_hif_edge_ids: set[HIF_ID] = set()
    for record in edge_records:
        hif_edge_id = record["edge"]
        edge_metadata = _get_record_metadata(record)
        edge_weight = record.get("weight")
        match h:
            case DirectedHypergraph():
                tail, head = directed_members_by_hif_edge_id.get(hif_edge_id, ([], []))
                directed_edge_key: DirectedEdge = (tuple(tail), tuple(head))
                h.add_edge(
                    directed_edge_key,
                    weight=edge_weight,
                    metadata=edge_metadata,
                )
            case Hypergraph():
                members = undirected_members_by_hif_edge_id.get(hif_edge_id)
                if members is None:
                    h.add_empty_edge(
                        hif_edge_id,
                        _get_record_metadata(record, include_weight=True),
                    )
                    continue
                undirected_edge_key: UndirectedEdge = tuple(members)
                h.add_edge(
                    undirected_edge_key,
                    weight=edge_weight,
                    metadata=edge_metadata,
                )
        added_hif_edge_ids.add(hif_edge_id)

    for incidence in data["incidences"]:
        hif_edge_id = incidence["edge"]
        node = incidence["node"]
        incidence_metadata = _get_record_metadata(incidence, include_weight=True)
        match h:
            case DirectedHypergraph():
                tail, head = directed_members_by_hif_edge_id[hif_edge_id]
                directed_edge_key = (tuple(tail), tuple(head))
                if hif_edge_id not in added_hif_edge_ids:
                    h.add_edge(directed_edge_key)
                    added_hif_edge_ids.add(hif_edge_id)
                if incidence_metadata:
                    h.set_incidence_metadata(
                        directed_edge_key, node, incidence_metadata
                    )
            case Hypergraph():
                undirected_edge_key = tuple(
                    undirected_members_by_hif_edge_id[hif_edge_id]
                )
                if hif_edge_id not in added_hif_edge_ids:
                    h.add_edge(undirected_edge_key)
                    added_hif_edge_ids.add(hif_edge_id)
                if incidence_metadata:
                    h.set_incidence_metadata(
                        undirected_edge_key, node, incidence_metadata
                    )
    return h


def read_hif(path: str) -> Hypergraph | DirectedHypergraph:
    """
    Load a hypergraph from a HIF file.

    Parameters
    ----------
    path : str
        The path to the HIF file

    Returns
    -------
    Hypergraph
        The loaded hypergraph
    """
    with open(path, encoding="utf-8") as file:
        data: HIFJson = json.load(file)
    return from_hif_dict(data)


def to_hif_dict(H: Hypergraph | DirectedHypergraph) -> HIFJson:
    """
    Create a dictionary following the HIF standard from a hypergraph.

    Parameters
    ----------
    H : Hypergraph or DirectedHypergraph
        The hypergraph to convert.

    Returns
    -------
    HIFJson
        A HIF dictionary containing the network type, metadata, nodes, edges,
        and incidences.

    Raises
    ------
    TypeError
        If ``H`` is not a supported hypergraph type or a metadata weight is not
        an integer or float.

    Notes
    -----
    Edge identifiers are generated for ordinary edges. Named empty edges in an
    undirected hypergraph retain their identifiers. All metadata is deeply
    copied into the returned dictionary.
    """
    match H:
        case DirectedHypergraph():
            network_type: NetworkType = "directed"
            empty_edges: dict[HIF_ID, Metadata] = {}
        case Hypergraph():
            network_type = "undirected"
            empty_edges = H.expose_data_structures().get("empty_edges", {})
        case _:
            raise TypeError(
                "HIF conversion supports Hypergraph and DirectedHypergraph objects."
            )

    data: HIFJson = {
        "network-type": network_type,
        "metadata": copy.deepcopy(H.get_hypergraph_metadata()),
        "edges": [],
        "nodes": [],
        "incidences": [],
    }

    for node, metadata in H.get_all_nodes_metadata().items():
        node_record: HIFNodeRecord = {"node": node}
        _add_metadata_in_record(node_record, metadata, include_weight=True)
        data["nodes"].append(node_record)

    hif_edge_ids = _get_hif_edge_ids(H.get_edge_list(), reserved_ids=set(empty_edges))
    incidence_metadata = H.get_all_incidences_metadata()
    for edge_key, hif_edge_id in hif_edge_ids.items():
        edge_record: HIFEdgeRecord = {"edge": hif_edge_id}
        _add_metadata_in_record(
            edge_record, H.get_edge_metadata(edge_key), include_weight=True
        )
        if H.is_weighted():
            edge_record["weight"] = _get_edge_weight(H, edge_key)
        data["edges"].append(edge_record)

        for node, direction in _get_hif_incidences(H, edge_key):
            incidence_record: HIFIncidenceRecord = {
                "edge": hif_edge_id,
                "node": node,
            }
            if direction is not None:
                incidence_record["direction"] = direction
            _add_metadata_in_record(
                incidence_record,
                incidence_metadata.get((edge_key, node), {}),
                include_weight=True,
            )
            data["incidences"].append(incidence_record)

    for hif_edge_id, metadata in empty_edges.items():
        edge_record: HIFEdgeRecord = {"edge": hif_edge_id}
        _add_metadata_in_record(edge_record, metadata, include_weight=True)
        data["edges"].append(edge_record)
    return data


def write_hif(H: Hypergraph | DirectedHypergraph, path: str) -> None:
    """
    Save a hypergraph to a HIF file.

    Parameters
    ----------
    H : Hypergraph or DirectedHypergraph
        The hypergraph to save.
    path : str
        The path to save the hypergraph to.
    """
    serialized_data = json.dumps(to_hif_dict(H), allow_nan=False)
    with open(path, "w", encoding="utf-8") as file:
        file.write(serialized_data)
