import json

from hypergraphx.core.undirected import Hypergraph
from hypergraphx.core.directed import DirectedHypergraph
from hypergraphx.core.multiplex import MultiplexHypergraph
from hypergraphx.core.temporal import TemporalHypergraph
from hypergraphx.exceptions import InvalidFormatError


def _split_json_records(data_list):
    hypergraph_metadata = {}
    hypergraph_type = None
    nodes = []
    edges = []

    for obj in data_list:
        if "hypergraph_metadata" in obj:
            hypergraph_metadata = obj["hypergraph_metadata"]
        if "hypergraph_type" in obj:
            hypergraph_type = obj["hypergraph_type"]
        t = obj.get("type")
        if t == "node":
            nodes.append(obj)
        elif t == "edge":
            edges.append(obj)

    return hypergraph_type, hypergraph_metadata, nodes, edges


def _build_hypergraph_from_json_objects(data_list):
    htype, meta, nodes, edges = _split_json_records(data_list)
    if htype not in {
        "Hypergraph",
        "DirectedHypergraph",
        "MultiplexHypergraph",
        "TemporalHypergraph",
    }:
        raise InvalidFormatError(f"Unsupported or missing 'hypergraph_type': {htype!r}")

    weighted = bool(meta.get("weighted", False))

    if htype == "Hypergraph":
        h = Hypergraph(hypergraph_metadata=meta, weighted=weighted)
    elif htype == "DirectedHypergraph":
        h = DirectedHypergraph(hypergraph_metadata=meta, weighted=weighted)
    elif htype == "MultiplexHypergraph":
        h = MultiplexHypergraph(hypergraph_metadata=meta, weighted=weighted)
    else:
        h = TemporalHypergraph(hypergraph_metadata=meta, weighted=weighted)

    for n in nodes:
        h.add_node(n["idx"], n["metadata"])

    if htype in {"Hypergraph", "DirectedHypergraph"}:
        for e in edges:
            interaction = e["interaction"]
            weight = e["metadata"].get("weight", None) if weighted else None
            h.add_edge(interaction, weight, metadata=e["metadata"])
    elif htype == "MultiplexHypergraph":
        for e in edges:
            interaction = e["interaction"]
            layer = e["metadata"].get("layer")
            weight = e["metadata"].get("weight", None) if weighted else None
            h.add_edge(interaction, layer, weight=weight, metadata=e["metadata"])
    else:
        for e in edges:
            interaction = e["interaction"]
            time = e["metadata"].get("time")
            weight = e["metadata"].get("weight", None) if weighted else None
            h.add_edge(interaction, time, weight=weight, metadata=e["metadata"])

    return h


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
