from hnx import Hypergraph
import json


def load_high_school(file_name: str, filter_by_class: list) -> Hypergraph:
    H = Hypergraph(weighted=False)
    with open(file_name, "r") as infile:
        data = json.load(infile)
        for obj in data:
            obj = eval(obj)
            if obj['type'] == 'node':
                try:
                    c = obj['class_school']
                except KeyError:
                    raise ValueError("Load the correct dataset")
                if c in filter_by_class:
                    H.add_node(obj['name'])
                    H.set_meta(obj['name'], obj)
            elif obj['type'] == 'edge':
                edge = tuple(sorted(obj['name']))
                add_edge = True
                for node in edge:
                    if node not in H.get_nodes():
                        add_edge = False
                        break
                if add_edge:
                    if H.is_weighted() or 'weight' in obj:
                        H._weighted = True
                    if not H.is_weighted():
                        H.add_edge(edge)
                    else:
                        H.add_edge(edge, obj['weight'])
                    H.set_meta(edge, obj)
    return H