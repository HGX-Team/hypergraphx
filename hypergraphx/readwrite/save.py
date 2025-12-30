from hypergraphx.readwrite.io_json import save_json_hypergraph
from hypergraphx.readwrite.io_pickle import save_pickle


def save_hypergraph(hypergraph, file_name: str, binary=False):
    if binary:
        save_pickle(hypergraph, file_name)
        return
    save_json_hypergraph(hypergraph, file_name)
