from __future__ import annotations

import warnings

from hypergraphx.readwrite.io_json import save_json_hypergraph
from hypergraphx.readwrite.io_pickle import save_pickle


def save_hypergraph(hypergraph, file_name: str, *, fmt: str = "json", binary=None):
    """
    Save a hypergraph to disk.

    Parameters
    ----------
    hypergraph
        Hypergraph-like object.
    file_name : str
        Output file path.
    fmt : {"json", "pickle"}
        Output format (default: "json").
    binary : bool | None
        Backward-compatible alias for `fmt="pickle"` when True.
        If provided, overrides `fmt` and emits a DeprecationWarning.
    """
    if binary is not None:
        warnings.warn(
            "save_hypergraph(..., binary=...) is deprecated; use fmt='pickle' or fmt='json'.",
            DeprecationWarning,
            stacklevel=2,
        )
        fmt = "pickle" if binary else "json"

    if fmt == "pickle":
        save_pickle(hypergraph, file_name)
        return
    if fmt == "json":
        save_json_hypergraph(hypergraph, file_name)
        return
    raise ValueError("fmt must be one of {'json', 'pickle'}.")
