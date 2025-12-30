import pickle

from hypergraphx.core.undirected import Hypergraph
from hypergraphx.core.directed import DirectedHypergraph
from hypergraphx.core.multiplex import MultiplexHypergraph
from hypergraphx.core.temporal import TemporalHypergraph
from hypergraphx.exceptions import ReadwriteError


def load_pickle(file_name):
    try:
        with open(file_name, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError("Pickle data is not a dictionary.")
        if "type" not in data:
            raise KeyError("The data is missing require key: 'type'.")

        h_type = data["type"]
        if h_type == "Hypergraph":
            h = Hypergraph(weighted=data["_weighted"])
        elif h_type == "TemporalHypergraph":
            h = TemporalHypergraph(weighted=data["_weighted"])
        elif h_type == "DirectedHypergraph":
            h = DirectedHypergraph(weighted=data["_weighted"])
        elif h_type == "MultiplexHypergraph":
            h = MultiplexHypergraph(weighted=data["_weighted"])
        else:
            raise ValueError(f"Unknown hypergraph type: {h_type}")

        h.populate_from_dict(data)
        return h
    except Exception as exc:
        raise ReadwriteError(f"Failed to load pickle '{file_name}': {exc}") from exc


def save_pickle(obj, file_name):
    try:
        if not hasattr(obj, "expose_data_structures"):
            raise AttributeError(
                "Object must implement 'expose_data_structures' method."
            )

        data = obj.expose_data_structures()

        with open(file_name, "wb") as f:
            pickle.dump(data, f)
    except Exception as exc:
        raise ReadwriteError(f"Failed to save object to {file_name}: {exc}") from exc
