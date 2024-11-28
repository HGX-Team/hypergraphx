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


def save_hypergraph(hypergraph, file_name: str, binary=False):
    """
    Save a hypergraph to a file.

    Parameters
    ----------
    hypergraph: Hypergraph
        The hypergraph to save
    file_name: str
        The requested name of the file
    binary: bool
        Whether to save the hypergraph as a binary file (hgx) or a text file (json)

    Returns
    -------
    None
        The hypergraph is saved to a file

    Raises
    ------
    ValueError
        If the file type is not valid

    Notes
    -----
    The file type can be either "pickle" or "json".
    """
    if binary:
        _save_pickle(hypergraph, file_name)
    else:
        with open(file_name, "w") as outfile:
            hypergraph_type = str(type(hypergraph)).split(".")[-1][:-2]
            out = []
            weighted = hypergraph.is_weighted()

            # Add hypergraph metadata

            d = {
                "hypergraph_type": hypergraph_type,
                "hypergraph_metadata": hypergraph.get_hypergraph_metadata(),
            }
            out.append(d)

            # Add nodes
            for node, metadata in hypergraph.get_nodes(metadata=True).items():
                d = {"type": "node", "idx": node, "metadata": metadata}
                out.append(d)

            # Add edges
            if hypergraph_type in ["Hypergraph", "DirectedHypergraph"]:
                for edge, metadata in hypergraph.get_edges(metadata=True).items():
                    d = {
                        "type": "edge",
                        "interaction": edge,
                        "metadata": metadata,
                    }
                    if weighted:
                        d["weight"] = hypergraph.get_weight(edge)
                    out.append(d)
            elif hypergraph_type == "MultiplexHypergraph":
                for edge, metadata in hypergraph.get_edges(metadata=True).items():
                    edge, layer = edge
                    d = {
                        "type": "edge",
                        "interaction": edge,
                        "metadata": metadata,
                        "layer": layer,
                    }
                    if weighted:
                        d["weight"] = hypergraph.get_weight(edge, layer)
                    out.append(d)
            elif hypergraph_type == "TemporalHypergraph":
                for edge, metadata in hypergraph.get_edges(metadata=True).items():
                    time, edge = edge
                    d = {
                        "type": "edge",
                        "interaction": edge,
                        "metadata": metadata,
                        "time": time,
                    }
                    if weighted:
                        d["weight"] = hypergraph.get_weight(edge, time)
                    out.append(d)
            else:
                raise ValueError("Invalid hypergraph type.")
            # Serialize and write the entire list to the file
            json.dump(out, outfile, indent=4)
