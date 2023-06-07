import json
import os
import pickle

from hypergraphx import Hypergraph


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


def load_hypergraph(file_name: str, file_type: str) -> Hypergraph:
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
    
    Notes
    -----
    The file type can be either "pickle", "json" or "hgr" (hmetis).
    """
    if file_type == "pickle":
        return _load_pickle(file_name)
    elif file_type == "json":
        H = Hypergraph(weighted=False)
        with open(file_name, "r") as infile:
            data = json.load(infile)
            for obj in data:
                obj = eval(obj)
                if obj['type'] == 'node':
                    H.add_node(obj['name'])
                    H.set_meta(obj['name'], obj)
                elif obj['type'] == 'edge':
                    if H.is_weighted() or 'weight' in obj:
                        H._weighted = True
                    if not H.is_weighted():
                        H.add_edge(tuple(sorted(obj['name'])))
                    else:
                        H.add_edge(tuple(sorted(obj['name'])), obj['weight'])
                    H.set_meta(tuple(sorted(obj['name'])), obj)
        return H
    elif file_type == "hgr":
        with open(file_name) as file:
            edges = 0
            nodes = 0
            mode = 0
            w_l = []
            edge_l = []
            read_count = 0
            read_node=0
            for line  in file:
                this_l = line.strip()
                if len(this_l)== 0 or this_l[0]=='%':
                    pass # do nothing for comments
                elif nodes ==0:
                    head = this_l.split(' ')
                    edges = int(head[0])
                    nodes  = int(head[1])
                    if len(head)==3:
                        mode = int(head[2])
                elif read_count<edges:
                    read_count += 1
                    entries = [int(r) for r in this_l.split(' ') if r != '']
                    if mode % 10 == 1 and len(entries)>1: # read weight
                        w_l += [int(entries[0])]
                        edge_l += [tuple(entries[1:])]
                    elif mode % 10 != 1 and len(entries)>0:
                        edge_l += [tuple(entries)]
                    else:
                        raise f"Empty edge in file. {read_count} edges read."
                elif read_node<nodes:
                    read_node += 1
                else:
                    raise f"File read to the end."
            H = Hypergraph(edge_list=edge_l,weighted=(mode % 10) == 1,weights=w_l if mode % 10 == 1 else None)
            return H
    else:
        raise ValueError("Invalid file type.")
