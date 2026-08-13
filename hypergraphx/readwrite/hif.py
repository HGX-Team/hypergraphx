import copy
import json
import math
import sys
from collections.abc import Hashable, Mapping
from os import PathLike
from typing import Any, Literal, TypeAlias, cast

from hypergraphx import DirectedHypergraph, Hypergraph

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:  # python 3.10 doesn't have NotRequired which is very useful for the HIF format
    from typing_extensions import NotRequired, TypedDict

__all__ = [
    "HIFSchema",
    "HIFEdgeRecord",
    "HIFIncidenceRecord",
    "HIFNodeRecord",
    "from_hif_dict",
    "read_hif",
    "to_hif_dict",
    "write_hif",
]

HIF_ID: TypeAlias = str | int
Weight: TypeAlias = int | float
Metadata: TypeAlias = dict[str, Any]
NetworkType: TypeAlias = Literal["asc", "directed", "undirected"]
Direction: TypeAlias = Literal["head", "tail"]
HypergraphType: TypeAlias = Hypergraph | DirectedHypergraph
UndirectedEdge: TypeAlias = tuple[Hashable, ...]
DirectedEdge: TypeAlias = tuple[tuple[Hashable, ...], tuple[Hashable, ...]]
EdgeKey: TypeAlias = UndirectedEdge | DirectedEdge
UndirectedMembers: TypeAlias = list[Hashable]
DirectedMembers: TypeAlias = tuple[list[Hashable], list[Hashable]]
EdgeMembers: TypeAlias = UndirectedMembers | DirectedMembers


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
HIFSchema = TypedDict(
    "HIFSchema",
    {
        "network-type": NotRequired[NetworkType],
        "metadata": NotRequired[Metadata],
        "incidences": list[HIFIncidenceRecord],
        "nodes": NotRequired[list[HIFNodeRecord]],
        "edges": NotRequired[list[HIFEdgeRecord]],
    },
)


def _record_metadata(
    record: Mapping[str, Any], include_weight: bool = False
) -> Metadata:
    """Copy the non-structural attributes from a HIF record."""
    attrs = copy.deepcopy(record.get("attrs", {}))
    if not isinstance(attrs, dict):
        raise TypeError("HIF record 'attrs' must be a dictionary.")
    if include_weight and "weight" in record:
        attrs["weight"] = record["weight"]
    return attrs


def _add_metadata(
    record: HIFNodeRecord | HIFEdgeRecord | HIFIncidenceRecord,
    metadata: Metadata,
    include_weight: bool = False,
) -> None:
    attrs = copy.deepcopy(metadata)
    if include_weight and "weight" in attrs:
        record["weight"] = attrs.pop("weight")
    if len(attrs) > 0:
        record["attrs"] = attrs


def _replace_nan_with_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _replace_nan_with_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_nan_with_none(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_nan_with_none(item) for item in value)
    return value


def _empty_edges(H: HypergraphType) -> dict[HIF_ID, Metadata]:
    """Copy empty-edge data exposed by HypergraphX's serialization API."""
    if not isinstance(H, Hypergraph):
        return {}
    return copy.deepcopy(H.expose_data_structures().get("empty_edges", {}))


def _edge_ids(H: HypergraphType) -> dict[EdgeKey, HIF_ID]:
    """Return HIF edge IDs which do not collide with named empty edges."""
    used_ids: set[HIF_ID] = set(_empty_edges(H))
    edge_ids: dict[EdgeKey, HIF_ID] = {}
    next_id = 0
    for edge, edge_id in H.get_edge_list().items():
        if edge_id in used_ids:
            while next_id in used_ids:
                next_id += 1
            edge_id = next_id
        edge_ids[edge] = edge_id
        used_ids.add(edge_id)
    return edge_ids


def _edge_key(members: EdgeMembers, directed: bool) -> EdgeKey:
    if directed:
        source, target = cast(DirectedMembers, members)
        return (tuple(source), tuple(target))
    return tuple(cast(UndirectedMembers, members))


def from_hif_dict(data: HIFSchema) -> HypergraphType:
    """Create a hypergraph from a dictionary following the HIF standard."""
    if not isinstance(data, dict):
        raise TypeError("HIF data must be provided as a dictionary.")
    network_type: NetworkType = data.get("network-type", "undirected")
    is_directed = network_type == "directed"
    metadata = copy.deepcopy(data.get("metadata", {}))
    if not isinstance(metadata, dict):
        raise TypeError("HIF 'metadata' must be a dictionary.")
    edge_records: list[HIFEdgeRecord] = data.get("edges", [])
    weighted = any("weight" in record for record in edge_records)

    if network_type == "undirected":
        H: HypergraphType = Hypergraph(weighted=weighted, duplicate_policy="error")
    elif is_directed:
        H = DirectedHypergraph(weighted=weighted, duplicate_policy="error")
    elif network_type == "asc":
        raise NotImplementedError(
            "HypergraphX does not support abstract simplicial complexes."
        )
    else:
        raise ValueError(f"Unknown hypergraph type: {network_type}")

    if "metadata" in data:
        H.set_hypergraph_metadata(metadata)

    tmp_edges: dict[Hashable, EdgeMembers] = {}
    for incidence in data["incidences"]:
        edge = incidence["edge"]
        node = incidence["node"]

        if edge not in tmp_edges:
            tmp_edges[edge] = ([], []) if is_directed else []
        if is_directed:
            directed_members = cast(DirectedMembers, tmp_edges[edge])
            direction = incidence.get("direction")
            if direction == "tail":
                directed_members[0].append(node)
            elif direction == "head":
                directed_members[1].append(node)
            else:
                raise ValueError(
                    "Directed HIF incidences require direction 'head' or 'tail'."
                )
        else:
            cast(UndirectedMembers, tmp_edges[edge]).append(node)

    for record in data.get("nodes", []):
        node = record["node"]
        H.add_node(node)
        H.set_node_metadata(node, _record_metadata(record, include_weight=True))

    added: dict[Hashable, EdgeKey] = {}

    for record in edge_records:
        edge = record["edge"]
        attrs = _record_metadata(record)
        if edge in tmp_edges:
            edge_key = _edge_key(tmp_edges[edge], is_directed)
            H.add_edge(edge_key, weight=record.get("weight"), metadata=attrs)
            added[edge] = edge_key
        elif is_directed:
            edge_key = ((), ())
            H.add_edge(edge_key, weight=record.get("weight"), metadata=attrs)
            added[edge] = edge_key
        else:
            if "weight" in record:
                attrs["weight"] = record["weight"]
            H.add_empty_edge(edge, attrs)

    for incidence in data["incidences"]:
        edge = incidence["edge"]
        node = incidence["node"]
        if edge not in added:
            edge_key = _edge_key(tmp_edges[edge], is_directed)
            H.add_edge(edge_key)
            added[edge] = edge_key
        attrs = _record_metadata(incidence, include_weight=True)
        if attrs:
            H.set_incidence_metadata(added[edge], node, attrs)

    return H


def read_hif(path: str | PathLike[str]) -> HypergraphType:
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
    edge_name_to_uid = {}
    node_name_to_uid = {}
    eid = 0
    nid = 0

    with open(path) as file:
        data: HIFSchema = json.loads(file.read())
    return from_hif_dict(data, nodetype=nodetype, edgetype=edgetype)

    if "type" not in data:
        logging.getLogger(__name__).warning("No hypergraph type - assume undirected")
        data["type"] = "undirected"

def to_hif_dict(H: HypergraphType, convert_nans: bool = False) -> HIFData:
    """Create a dictionary following the HIF standard from a hypergraph."""
    if isinstance(H, DirectedHypergraph):
        network_type: NetworkType = "directed"
    elif isinstance(H, Hypergraph):
        network_type = "undirected"
    else:
        raise TypeError(
            "HIF conversion supports Hypergraph and DirectedHypergraph objects."
        )

    data: HIFData = {
        "network-type": network_type,
        "metadata": copy.deepcopy(H.get_hypergraph_metadata()),
        "edges": [],
        "nodes": [],
        "incidences": [],
    }

    for node, attrs in H.get_all_nodes_metadata().items():
        node_record: HIFNodeRecord = {"node": node}
        _add_metadata(node_record, attrs, include_weight=True)
        data["nodes"].append(node_record)

        if incidence["node"] not in node_name_to_uid:
            node_name_to_uid[incidence["node"]] = nid
            nid += 1
        node = node_name_to_uid[incidence["node"]]

        if edge not in tmp_edges:
            tmp_edges[edge] = []
        tmp_edges[edge].append(node)

    for record in data["nodes"]:
        node_name = record["node"]
        if node_name not in node_name_to_uid:
            node_name_to_uid[node_name] = nid
            nid += 1
        node = node_name_to_uid[node_name]
        H.add_node(node)
        H.set_node_metadata(node, record)

    added = {}

    for record in data["edges"]:
        edge_name = record["edge"]
        if edge_name not in edge_name_to_uid:
            edge_name_to_uid[edge_name] = eid
            eid += 1
        edge = edge_name_to_uid[edge_name]
        if edge in tmp_edges:
            H.add_edge(tuple(sorted(tmp_edges[edge])))
            added[tuple(sorted(tmp_edges[edge]))] = True
            H.set_edge_metadata(tuple(sorted(tmp_edges[edge])), record)
        else:
            H.add_empty_edge(edge_name, record)

    for incidence in data["incidences"]:
        edge = edge_name_to_uid[incidence["edge"]]
        node = node_name_to_uid[incidence["node"]]
        if tuple(sorted(tmp_edges[edge])) not in added:
            H.add_edge(tuple(sorted(tmp_edges[edge])))
            added[tuple(sorted(tmp_edges[edge]))] = True
        H.set_incidence_metadata(tuple(sorted(tmp_edges[edge])), node, incidence)

    return H


def write_hif(H: HypergraphType, path: str) -> None:
    """
    Save a hypergraph to a HIF file.

    Parameters
    ----------
    H: Hypergraph
        The hypergraph to save.
    path: str
        The path to save the hypergraph to.
    """

    data = {
        "type": "undirected",
        "metadata": H.get_hypergraph_metadata(),
        "edges": H.get_all_edges_metadata(),
        "nodes": H.get_all_nodes_metadata(),
        "incidences": H.get_all_incidences_metadata(),
    }

    with open(path, "w") as file:
        file.write(json.dumps(to_hif_dict(H, convert_nans=True)))
