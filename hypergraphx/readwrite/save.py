import json
import pickle

from hypergraphx import (
    Hypergraph,
    TemporalHypergraph,
    DirectedHypergraph,
    MultiplexHypergraph,
)

import pickle


def _save_pickle(obj, file_name: str):
    """
    Save an object as a pickle file.

    Parameters
    ----------
    obj : object
        The object to save. Must implement `expose_data_structures`.
    file_name : str
        The name of the file to save the object to.

    Returns
    -------
    None
        The object is saved to a file.
    """
    try:
        if not hasattr(obj, "expose_data_structures"):
            raise AttributeError(
                "Object must implement 'expose_data_structures' method."
            )

        data = obj.expose_data_structures()

        with open(file_name, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        raise RuntimeError(f"Failed to save object to {file_name}: {e}")


import json


def save_hypergraph(hypergraph, file_name: str, binary=False):
    """
    Save a hypergraph to a file in JSON format efficiently by writing in chunks.

    Parameters
    ----------
    hypergraph: Hypergraph
        The hypergraph to save.
    file_name: str
        The name of the file.
    binary: bool
        Whether to save the hypergraph as a binary file (hgx) or a text file (json).

    Returns
    -------
    None
        The hypergraph is saved to a file.

    Raises
    ------
    ValueError
        If the hypergraph type is not valid.
    """
    if binary:
        _save_pickle(hypergraph, file_name)
        return

    with open(file_name, "w") as outfile:
        outfile.write("[\n")  # Start JSON array
        first = True

        def write_item(item):
            """Helper function to write a JSON object, ensuring proper comma placement."""
            nonlocal first
            if not first:
                outfile.write(",\n")
            json.dump(
                item, outfile, separators=(",", ":")
            )  # No pretty-print for efficiency
            first = False

        # Get hypergraph metadata
        hypergraph_type = str(type(hypergraph)).split(".")[-1][:-2]
        weighted = hypergraph.is_weighted()

        write_item(
            {
                "hypergraph_type": hypergraph_type,
                "hypergraph_metadata": hypergraph.get_hypergraph_metadata(),
            }
        )

        # Write nodes
        for node, metadata in hypergraph.get_nodes(metadata=True).items():
            write_item({"type": "node", "idx": node, "metadata": metadata})

        # Write edges
        if hypergraph_type in ["Hypergraph", "DirectedHypergraph"]:
            for edge, metadata in hypergraph.get_edges(metadata=True).items():
                if weighted:
                    metadata["weight"] = hypergraph.get_weight(edge)
                write_item({"type": "edge", "interaction": edge, "metadata": metadata})

        elif hypergraph_type == "MultiplexHypergraph":
            for edge, metadata in hypergraph.get_edges(metadata=True).items():
                edge, layer = edge
                metadata["layer"] = layer
                if weighted:
                    metadata["weight"] = hypergraph.get_weight(edge, layer)
                write_item({"type": "edge", "interaction": edge, "metadata": metadata})

        elif hypergraph_type == "TemporalHypergraph":
            for edge, metadata in hypergraph.get_edges(metadata=True).items():
                time, edge = edge
                if weighted:
                    metadata["weight"] = hypergraph.get_weight(edge, time)
                metadata["time"] = time
                write_item({"type": "edge", "interaction": edge, "metadata": metadata})

        else:
            raise ValueError("Invalid hypergraph type.")

        outfile.write("\n]")  # Close JSON array
