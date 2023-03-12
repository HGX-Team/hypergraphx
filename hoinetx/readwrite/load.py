import pickle
import json
from hoinetx.core.hypergraph import Hypergraph


def _load_pickle(file_name):
    with open("{}".format(file_name), "rb") as f:
        return pickle.load(f)


def load_hypergraph(file_name, file_type):
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
