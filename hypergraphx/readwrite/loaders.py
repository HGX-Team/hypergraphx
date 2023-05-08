import json

from hypergraphx import Hypergraph


def load_high_school(file_name: str, filter_by_class=None) -> Hypergraph:
    """
    Load the high school dataset from the file_name and apply filters

    Parameters
    ----------
    file_name: str
        The path of the high school dataset
    filter_by_class: list of str or None
        If not None, only the classes in the list will be loaded

    Returns
    -------
    H: Hypergraph
        The loaded hypergraph

    Raises
    ------
    ValueError
        If the dataset is not the correct one
    """

    classIDs = {}
    next_class_id = 0
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
                if c not in classIDs:
                    classIDs[c] = next_class_id
                    next_class_id += 1
                if filter_by_class is None or c in filter_by_class:
                    H.add_node(obj['name'])
                    H.set_meta(obj['name'], obj)
                    H.add_attr_meta(obj['name'], 'classID', classIDs[c])
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


#H = load_high_school("../../test_data/hs/hs.json")
#print(H.get_meta(151))
