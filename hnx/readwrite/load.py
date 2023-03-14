import pickle
import json
import os
from hnx.core.hypergraph import Hypergraph


def _load_pickle(file_name: str) -> Hypergraph:
    """
    Load a pickle file.

    Parameters
    ----------
    file_name: str
        Name of the file

    Returns
    -------
    Hypergraph
        The loaded hypergraph
    """
    with open("{}".format(file_name), "rb") as f:
        return pickle.load(f)


def _check_existence(file_name: str, file_type: str):
    """
    Check if a file exists.
    Parameters
    ----------
    file_name : str
        Name of the file
    file_type : str
        Type of the file

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    file_name += ".{}".format(file_type)
    return os.path.isfile(file_name)


def load_hypergraph(file_name: str, file_type: str):
    """
    Load a hypergraph from a file.

    Parameters
    ----------
    file_name : str
        The name of the file
    file_type : str
        The type of the file

    Returns
    -------
    Hypergraph
        The loaded hypergraph

    Raises
    ------
    ValueError
        If the file type is not valid.
    """
    file_name += ".{}".format(file_type)
    if file_type == "pickle":
        return _load_pickle(file_name)
    elif file_type == "json":
        H = Hypergraph()
        with open(file_name, "r") as infile:
            data = json.load(infile)
            for obj in data:
                obj = eval(obj)
                if obj['type'] == 'node':
                    H.add_node(obj['name'])
                    H.set_meta(obj['name'], obj)
                elif obj['type'] == 'edge':
                    H.add_edge(tuple(sorted(obj['name'])))
                    if 'weight' in obj:
                        H.set_weight(tuple(sorted(obj['name'])), obj['weight'])
                    H.set_meta(tuple(sorted(obj['name'])), obj)
        return H
    else:
        raise ValueError("Invalid file type.")
