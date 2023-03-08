import pickle


def load_pickle(file_name):
    with open("{}".format(file_name), "rb") as f:
        return pickle.load(f)
