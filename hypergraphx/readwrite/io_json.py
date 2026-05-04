import json

from hypergraphx.core.undirected import Hypergraph
from hypergraphx.core.directed import DirectedHypergraph
from hypergraphx.core.multiplex import MultiplexHypergraph
from hypergraphx.core.temporal import TemporalHypergraph
from hypergraphx.exceptions import InvalidFormatError
from hypergraphx.exceptions import InvalidParameterError
from hypergraphx.utils.metadata import merge_metadata


def _get_json_header(data_list):
    htype = None
    meta = {}
    for obj in data_list:
        if "hypergraph_metadata" in obj:
            meta = obj["hypergraph_metadata"]
        if "hypergraph_type" in obj:
            htype = obj["hypergraph_type"]
            break

    if htype not in {
        "Hypergraph",
        "DirectedHypergraph",
        "MultiplexHypergraph",
        "TemporalHypergraph",
    }:
        raise InvalidFormatError(f"Unsupported or missing 'hypergraph_type': {htype!r}")

    return htype, meta


def _new_hypergraph_for_json(htype, meta, weighted):
    if htype == "Hypergraph":
        return Hypergraph(hypergraph_metadata=meta, weighted=weighted)
    if htype == "DirectedHypergraph":
        return DirectedHypergraph(hypergraph_metadata=meta, weighted=weighted)
    if htype == "MultiplexHypergraph":
        return MultiplexHypergraph(hypergraph_metadata=meta, weighted=weighted)
    return TemporalHypergraph(hypergraph_metadata=meta, weighted=weighted)


def _validate_json_weight(weight, weighted):
    if not weighted and weight is not None and weight != 1:
        raise ValueError("If the hypergraph is not weighted, weight can be 1 or None.")
    return 1 if weight is None else weight


def _apply_duplicate_json_edge(
    edge_key,
    weight,
    metadata,
    *,
    weighted,
    duplicate_policy,
    metadata_policy,
    edge_list,
    weights,
    edge_metadata,
):
    if not weighted and duplicate_policy in {"accumulate_weight", "replace_weight"}:
        raise InvalidParameterError(
            "duplicate_policy must be 'ignore' or 'error' for unweighted hypergraphs."
        )

    edge_id = edge_list[edge_key]
    if duplicate_policy == "error":
        raise InvalidParameterError(f"Duplicate edge {edge_key} not allowed.")
    if duplicate_policy == "ignore":
        pass
    elif duplicate_policy == "accumulate_weight":
        if weighted:
            weights[edge_id] += weight
    elif duplicate_policy == "replace_weight":
        if weighted:
            weights[edge_id] = weight
    else:
        raise InvalidParameterError(
            "duplicate_policy must be one of: 'error', 'ignore', 'accumulate_weight', 'replace_weight'."
        )

    if not metadata:
        return
    if metadata_policy == "replace":
        edge_metadata[edge_id] = metadata
    elif metadata_policy == "merge":
        edge_metadata[edge_id] = merge_metadata(edge_metadata.get(edge_id), metadata)
    elif metadata_policy == "ignore":
        pass
    else:
        raise InvalidParameterError(
            "metadata_policy must be one of: 'replace', 'merge', 'ignore'."
        )


def _assign_json_structures(
    h,
    htype,
    *,
    node_metadata,
    edge_metadata,
    edge_list,
    reverse_edge_list,
    weights,
    next_edge_id,
    adj=None,
    adj_source=None,
    adj_target=None,
    existing_layers=None,
):
    h._node_metadata = node_metadata
    h._edge_metadata = edge_metadata
    h._edge_list = edge_list
    h._reverse_edge_list = reverse_edge_list
    h._weights = weights
    h._incidences_metadata = {}
    h._next_edge_id = next_edge_id
    if htype == "DirectedHypergraph":
        h._adj_source = adj_source
        h._adj_target = adj_target
    else:
        h._adj = adj
    if htype == "MultiplexHypergraph":
        h._existing_layers = existing_layers or set()
    h._maybe_validate_invariants()
    return h


def _build_hypergraph_from_json_objects(data_list):
    htype, meta = _get_json_header(data_list)
    weighted = bool(meta.get("weighted", False))
    h = _new_hypergraph_for_json(htype, meta, weighted)

    node_metadata = {}
    edge_metadata = {}
    edge_list = {}
    reverse_edge_list = {}
    weights = {}
    next_edge_id = 0
    duplicate_policy = h._duplicate_policy
    metadata_policy = h._metadata_policy
    existing_layers = set()

    if htype == "DirectedHypergraph":
        adj = None
        adj_source = {}
        adj_target = {}

        def ensure_node(node, metadata=None):
            if metadata is None:
                metadata = {}
            if node not in node_metadata:
                adj_source[node] = []
                adj_target[node] = []
                node_metadata[node] = metadata
            elif node_metadata[node] == {} and metadata:
                node_metadata[node] = metadata

        def add_incidence(edge_key, edge_id):
            source, target = edge_key
            for node in source:
                ensure_node(node)
                adj_source[node].append(edge_id)
            for node in target:
                ensure_node(node)
                adj_target[node].append(edge_id)

    else:
        adj = {}
        adj_source = None
        adj_target = None

        def ensure_node(node, metadata=None):
            if metadata is None:
                metadata = {}
            if node not in node_metadata:
                adj[node] = []
                node_metadata[node] = metadata
            elif node_metadata[node] == {} and metadata:
                node_metadata[node] = metadata

        def add_incidence(edge_key, edge_id):
            for node in h._edge_nodes(edge_key):
                ensure_node(node)
                adj[node].append(edge_id)

    def normalize_edge(interaction, metadata):
        if htype in {"Hypergraph", "DirectedHypergraph"}:
            return h._normalize_edge(interaction)
        if htype == "MultiplexHypergraph":
            return h._normalize_edge(interaction, layer=metadata.get("layer"))
        return h._normalize_edge(interaction, time=metadata.get("time"))

    def add_edge(edge_key, weight, metadata):
        nonlocal next_edge_id
        weight = _validate_json_weight(weight, weighted)
        metadata = metadata or {}
        if htype == "MultiplexHypergraph":
            existing_layers.add(edge_key[0])

        if edge_key in edge_list:
            _apply_duplicate_json_edge(
                edge_key,
                weight,
                metadata,
                weighted=weighted,
                duplicate_policy=duplicate_policy,
                metadata_policy=metadata_policy,
                edge_list=edge_list,
                weights=weights,
                edge_metadata=edge_metadata,
            )
            return

        edge_id = next_edge_id
        next_edge_id += 1
        edge_list[edge_key] = edge_id
        reverse_edge_list[edge_id] = edge_key
        weights[edge_id] = 1 if not weighted else weight
        edge_metadata[edge_id] = metadata
        add_incidence(edge_key, edge_id)

    for obj in data_list:
        record_type = obj.get("type")
        if record_type == "node":
            ensure_node(obj["idx"], obj.get("metadata") or {})
        elif record_type == "edge":
            metadata = obj.get("metadata") or {}
            edge_key = normalize_edge(obj["interaction"], metadata)
            weight = metadata.get("weight", None) if weighted else None
            add_edge(edge_key, weight, metadata)

    return _assign_json_structures(
        h,
        htype,
        node_metadata=node_metadata,
        edge_metadata=edge_metadata,
        edge_list=edge_list,
        reverse_edge_list=reverse_edge_list,
        weights=weights,
        next_edge_id=next_edge_id,
        adj=adj,
        adj_source=adj_source,
        adj_target=adj_target,
        existing_layers=existing_layers,
    )


def _parse_json_bytes_to_hypergraph(data):
    try:
        data_list = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise InvalidFormatError("Failed to parse JSON payload.") from exc
    return _build_hypergraph_from_json_objects(data_list)


def load_json_file(file_name):
    with open(file_name, "r", encoding="utf-8") as infile:
        data_list = json.load(infile)
    return _build_hypergraph_from_json_objects(data_list)


def save_json_hypergraph(hypergraph, file_name):
    with open(file_name, "w") as outfile:
        outfile.write("[\n")
        first = True

        def write_item(item):
            nonlocal first
            if not first:
                outfile.write(",\n")
            json.dump(item, outfile, separators=(",", ":"))
            first = False

        hypergraph_type = str(type(hypergraph)).split(".")[-1][:-2]
        weighted = hypergraph.is_weighted()

        write_item(
            {
                "hypergraph_type": hypergraph_type,
                "hypergraph_metadata": hypergraph.get_hypergraph_metadata(),
            }
        )

        for node, metadata in hypergraph.get_nodes(metadata=True).items():
            write_item({"type": "node", "idx": node, "metadata": metadata})

        if hypergraph_type in ["Hypergraph", "DirectedHypergraph"]:
            for edge, metadata in hypergraph.get_edges(metadata=True).items():
                metadata = dict(metadata)
                if weighted:
                    metadata["weight"] = hypergraph.get_weight(edge)
                write_item({"type": "edge", "interaction": edge, "metadata": metadata})
        elif hypergraph_type == "MultiplexHypergraph":
            for edge, metadata in hypergraph.get_edges(metadata=True).items():
                metadata = dict(metadata)
                layer, edge = edge
                metadata["layer"] = layer
                if weighted:
                    metadata["weight"] = hypergraph.get_weight(edge, layer)
                write_item({"type": "edge", "interaction": edge, "metadata": metadata})
        elif hypergraph_type == "TemporalHypergraph":
            for edge, metadata in hypergraph.get_edges(metadata=True).items():
                metadata = dict(metadata)
                time, edge = edge
                if weighted:
                    metadata["weight"] = hypergraph.get_weight(edge, time)
                metadata["time"] = time
                write_item({"type": "edge", "interaction": edge, "metadata": metadata})
        else:
            raise ValueError("Invalid hypergraph type.")

        outfile.write("\n]")
