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


def save_hypergraph(hypergraph: Hypergraph, file_name: str, file_type = 'json'):
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
        with open(file_name + '.' + file_type, "w+") as outfile:
            out = []
            d = {}
            d['hypergraph_metadata'] = hypergraph.get_hypergraph_metadata()
            json_object = json.dumps(d)
            out.append(json_object)
            
            for node in hypergraph.get_nodes():
                d = {}
                d['type'] = 'node'
                d['idx'] = node
                d['metadata'] = hypergraph.get_node_metadata(node)
                json_object = json.dumps(d)
                out.append(json_object)

            for edge in hypergraph.get_edges():
                d = {}
                d['type'] = 'edge'
                d['interaction'] = edge
                d['metadata'] = hypergraph.get_edge_metadata(edge)
                if hypergraph.is_weighted():
                    d["weight"] = hypergraph.get_weight(edge)
                json_object = json.dumps(d)
                out.append(json_object)
            json.dump(out, outfile)

    else:
        raise ValueError("Invalid file type.")

