import json
import pickle

from hypergraphx import Hypergraph


def _save_pickle(obj, file_name: str):
    """
    Save a pickle file.

    Parameters
    ----------
    obj : object
        The object to save
    file_name : str
        The name of the file

    Returns
    -------
    None
        the object is saved to a file
    """
    with open("{}".format(file_name), "wb") as f:
        pickle.dump(obj, f)


def save_hypergraph(hypergraph: Hypergraph, file_name: str, file_type: str):
    """
    Save a hypergraph to a file.

    Parameters
    ----------
    hypergraph: Hypergraph
        The hypergraph to save
    file_name: str
        The requested name of the file
    file_type: str
        The requested type of the file

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
    if file_type == "pickle":
        _save_pickle(hypergraph, file_name)
    elif file_type == "json":
        with open(file_name, "w+") as outfile:
            out = []
            for node in hypergraph.get_nodes():
                json_object = json.dumps(hypergraph.get_meta(node))
                out.append(json_object)

            for edge in hypergraph.get_edges():
                meta = hypergraph.get_meta(edge)
                if hypergraph.is_weighted():
                    meta["weight"] = hypergraph.get_weight(edge)
                json_object = json.dumps(meta)
                out.append(json_object)
            json.dump(out, outfile)

    else:
        raise ValueError("Invalid file type.")

