import pickle
from hoinetx.core.hypergraph import Hypergraph


def save_pickle(obj, file_name):
    with open("{}".format(file_name), "wb") as f:
        pickle.dump(obj, f)


def save(hypergraph: Hypergraph, file_name, file_type="pickle"):

    if file_type == "pickle":
        with open(file_name, 'wb') as file:
            pickle.dump(hypergraph, file)
    elif file_type == "json":
        pass
    elif file_type == "text":
        pass
    elif file_type == "csv":
        pass
    else:
        raise ValueError("Invalid file type.")


def load(file_name, file_type="pickle"):

    if file_type == "pickle":
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    elif file_type == "json":
        pass
    elif file_type == "text":
        pass
    elif file_type == "csv":
        pass
    else:
        raise ValueError("Invalid file type.")

