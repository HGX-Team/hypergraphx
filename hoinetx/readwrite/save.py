import pickle
from hoinetx.core.hypergraph import Hypergraph


def _save_pickle(obj, file_name):
    with open("{}".format(file_name), "wb") as f:
        pickle.dump(obj, f)


def save_hypergraph(hypergraph: Hypergraph, file_name, file_type="pickle"):
    if file_type == "pickle":
        _save_pickle(hypergraph, file_name)
    elif file_type == "json":
        pass
    elif file_type == "text":
        pass
    elif file_type == "csv":
        pass
    else:
        raise ValueError("Invalid file type.")

