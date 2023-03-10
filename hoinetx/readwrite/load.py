import pickle, json
from hoinetx.core.hypergraph import Hypergraph


def _load_pickle(file_name):
    with open("{}".format(file_name), "rb") as f:
        return pickle.load(f)


def load_hypergraph(file_name, file_type):
    file_name += ".{}".format(file_type)
    if file_type == "pickle":
        _load_pickle(file_name)
    elif file_type == "json":
        with open(file_name, "r") as infile:
            studentDict = json.load(infile)
            print(studentDict)

    else:
        raise ValueError("Invalid file type.")
