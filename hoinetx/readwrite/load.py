import pickle


def _load_pickle(file_name):
    with open("{}".format(file_name), "rb") as f:
        return pickle.load(f)


def load_hypergraph(file_name, file_type="pickle"):
    if file_type == "pickle":
        _load_pickle(file_name)
    elif file_type == "json":
        pass
    elif file_type == "text":
        pass
    elif file_type == "csv":
        pass
    else:
        raise ValueError("Invalid file type.")
