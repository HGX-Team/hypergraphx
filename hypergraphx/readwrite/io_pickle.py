import pickle

from hypergraphx.core.undirected import Hypergraph
from hypergraphx.core.directed import DirectedHypergraph
from hypergraphx.core.multiplex import MultiplexHypergraph
from hypergraphx.core.temporal import TemporalHypergraph
from hypergraphx.exceptions import ReadwriteError

# Globals a restricted load may reference: plain containers and scalars, plus
# numpy scalars/arrays (numpy 1.x and 2.x module paths). Hypergraph pickles
# written by save_pickle only need builtins, which reference no globals at all.
_SAFE_PICKLE_GLOBALS = frozenset(
    [
        ("builtins", name)
        for name in (
            "bool",
            "bytearray",
            "bytes",
            "complex",
            "dict",
            "float",
            "frozenset",
            "int",
            "list",
            "range",
            "set",
            "slice",
            "str",
            "tuple",
        )
    ]
    + [
        ("collections", name)
        for name in ("Counter", "OrderedDict", "defaultdict", "deque")
    ]
    + [
        ("datetime", name)
        for name in ("date", "datetime", "time", "timedelta", "timezone")
    ]
    + [("numpy", "dtype"), ("numpy", "ndarray")]
    + [
        (module, name)
        for module in ("numpy.core.multiarray", "numpy._core.multiarray")
        for name in ("scalar", "_reconstruct")
    ]
    + [
        (module, "_frombuffer")
        for module in ("numpy.core.numeric", "numpy._core.numeric")
    ]
)


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _SAFE_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(
                f"Refusing to load '{module}.{name}' from an untrusted pickle."
            )
        return super().find_class(module, name)


def load_pickle(file_name, *, restricted=False):
    """
    Load a hypergraph saved with :func:`save_pickle`.

    Parameters
    ----------
    file_name : str
        Path to the pickle file.
    restricted : bool, default=False
        If True, only plain data types (containers, scalars, numpy arrays) may
        be unpickled, so a malicious file cannot execute code. Use this for
        files from untrusted sources; remote datasets are always loaded this
        way.
    """
    try:
        with open(file_name, "rb") as f:
            if restricted:
                data = _RestrictedUnpickler(f).load()
            else:
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
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise ReadwriteError(f"Failed to save object to {file_name}: {exc}") from exc
