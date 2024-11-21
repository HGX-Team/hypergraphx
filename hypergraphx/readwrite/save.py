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


import json

def save_hypergraph(hypergraph, file_name: str, file_type='json'):
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
        with open(file_name + '.' + file_type, "w") as outfile:
            hypergraph_type = str(type(hypergraph)).split('.')[-1][:-2]
            out = []

            # Add hypergraph metadata

            d = {
                'hypergraph_type': hypergraph_type,
                'hypergraph_metadata': hypergraph.get_hypergraph_metadata()
            }
            out.append(d)

            # Add nodes
            for node, metadata in hypergraph.get_nodes(metadata=True).items():
                d = {
                    'type': 'node',
                    'idx': node,
                    'metadata': metadata
                }
                out.append(d)

            # Add edges
            if hypergraph_type in ['Hypergraph', 'DirectedHypergraph']:
                for edge, metadata in hypergraph.get_edges(metadata=True).items():
                    d = {
                        'type': 'edge',
                        'interaction': edge,
                        'metadata': metadata,
                        'weight': hypergraph.get_weight(edge)
                    }
                    out.append(d)
            elif hypergraph_type == 'MultiplexHypergraph':
                for edge, metadata in hypergraph.get_edges(metadata=True).items():
                    edge, layer = edge
                    d = {
                        'type': 'edge',
                        'interaction': edge,
                        'metadata': metadata,
                        'weight': hypergraph.get_weight(edge, layer),
                        'layer': layer
                    }
                    out.append(d)

            # Serialize and write the entire list to the file
            json.dump(out, outfile, indent=4)

    else:
        raise ValueError("Invalid file type.")


