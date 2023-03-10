import pickle, os, json
from hoinetx.core.hypergraph import Hypergraph


def _save_pickle(obj, file_name):
    with open("{}".format(file_name), "wb") as f:
        pickle.dump(obj, f)


def save_hypergraph(hypergraph: Hypergraph, file_name, file_type):
    if file_type == "pickle":
        _save_pickle(hypergraph, file_name)
    elif file_type == "json":
        file_name += ".json"
        with open(file_name, "w+") as outfile:
            out = []
            for node in hypergraph.get_nodes():
                json_object = json.dumps(hypergraph.get_meta(node))
                out.append(json_object)

            for edge in hypergraph.get_edges():
                meta = hypergraph.get_meta(edge)
                if hypergraph.is_weighted():
                    meta["weight"] = hypergraph.get_weight(edge)
                json_object = json.dumps(meta)
                out.append(json_object)
            json.dump(out, outfile)

    else:
        raise ValueError("Invalid file type.")

