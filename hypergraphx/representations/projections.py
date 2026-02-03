import networkx as nx
from hypergraphx import Hypergraph, DirectedHypergraph
from hypergraphx.measures.edge_similarity import intersection, jaccard_similarity


def bipartite_projection(
    h: Hypergraph,
    *,
    node_order=None,
    edge_order=None,
    return_obj_to_id: bool = False,
):
    """
    Returns a bipartite graph representation of the hypergraph.

    Parameters
    ----------
    h : Hypergraph
        The hypergraph to be projected.
    node_order : list, optional (keyword-only)
        Explicit node iteration order to make the node-id mapping deterministic.
        If None, uses `h.get_nodes()` order.
    edge_order : list, optional (keyword-only)
        Explicit edge iteration order to make the edge-id mapping deterministic.
        If None, uses `h.get_edges()` order.
    return_obj_to_id : bool, optional (keyword-only)
        If True, also return the reverse mapping `obj_to_id`.

    Returns
    -------
    tuple
        `(g, id_to_obj)` where:
        - `g` is a `networkx.Graph` with node ids like `"N0"` and `"E0"`
        - `id_to_obj` maps node ids to original objects (node labels and edge tuples)

        If `return_obj_to_id=True`, returns `(g, id_to_obj, obj_to_id)`.

    Notes
    -----
    This function is deterministic given `node_order` and `edge_order`.
    Without them, the mapping depends on the insertion/iteration order of `h`.
    """
    g = nx.Graph()
    id_to_obj = {}
    obj_to_id = {}
    idx = 0

    nodes = h.get_nodes() if node_order is None else list(node_order)
    for node in nodes:
        id_to_obj["N" + str(idx)] = node
        obj_to_id[node] = "N" + str(idx)
        idx += 1
        g.add_node(obj_to_id[node], bipartite=0)

    idx = 0

    edges = h.get_edges() if edge_order is None else list(edge_order)
    for edge in edges:
        edge_key = h._normalize_edge(edge)
        obj_to_id[edge_key] = "E" + str(idx)
        id_to_obj["E" + str(idx)] = edge_key
        idx += 1
        g.add_node(obj_to_id[edge_key], bipartite=1)

        for node in edge_key:
            g.add_edge(obj_to_id[edge_key], obj_to_id[node])

    if return_obj_to_id:
        return g, id_to_obj, obj_to_id
    return g, id_to_obj


def clique_projection(h: Hypergraph, keep_isolated=False):
    """
    Returns a clique projection of the hypergraph.

    Parameters
    ----------
    h : Hypergraph
        The hypergraph to be projected.
    keep_isolated : bool
        Whether to keep isolated nodes or not.

    Returns
    -------
    networkx.Graph
        The clique projection of the hypergraph.

    Notes
    -----
    Computing the clique projection can be very expensive for large hypergraphs.

    Example
    -------
    >>> import networkx as nx
    >>> import hypergraphx as hgx
    >>> from hypergraphx.representations.projections import clique_projection
    >>>
    >>> h = hgx.Hypergraph()
    >>> h.add_nodes([1, 2, 3, 4, 5])
    >>> h.add_edges([(1, 2), (1, 2, 3), (3, 4, 5)])
    >>> g = clique_projection(h)
    >>> g.edges()
    EdgeView([(1, 2), (1, 3), (2,3), (3, 4), (3, 5), (4, 5)])
    """
    g = nx.Graph()

    if keep_isolated:
        for node in h.get_nodes():
            g.add_node(node)

    for edge in h.get_edges():
        for i in range(len(edge) - 1):
            for j in range(i + 1, len(edge)):
                g.add_edge(edge[i], edge[j])

    return g


def line_graph(
    h: Hypergraph,
    distance="intersection",
    s=1,
    weighted=False,
    *,
    edge_order=None,
):
    """
    Returns a line graph of the hypergraph.

    Parameters
    ----------
    h : Hypergraph
        The hypergraph to be projected.
    distance : str
        The distance function to be used. Can be 'intersection' or 'jaccard'.
    s : float
        The threshold for the distance function.
    weighted : bool
        Whether the line graph should be weighted or not.
    edge_order : list, optional (keyword-only)
        Explicit edge iteration order to make the returned `id_to_edge` mapping deterministic.
        If None, uses `h.get_edges()` order.

    Returns
    -------
    tuple
        `(g, id_to_edge)` where:
        - `g` is a `networkx.Graph` whose nodes are edge-ids `0..m-1`
        - `id_to_edge` maps those ids back to hyperedges

    Notes
    -----
    Computing the line graph can be very expensive for large hypergraphs.
    This function is deterministic given `edge_order`. Without it, edge-id assignment
    depends on the insertion/iteration order of `h`.

    Example
    -------
    >>> import networkx as nx
    >>> import hypergraphx as hgx
    >>> from hypergraphx.representations.projections import line_graph
    >>>
    >>> h = hgx.Hypergraph()
    >>> h.add_nodes([1, 2, 3, 4, 5])
    >>> h.add_edges([(1, 2), (1, 2, 3), (3, 4, 5)])
    >>> g, idx = line_graph(h)
    >>> g.edges()
    EdgeView([(0, 1), (1, 2)])
    """

    def _distance(a, b):
        if distance == "intersection":
            return intersection(a, b)
        if distance == "jaccard":
            return jaccard_similarity(a, b)

    edges = h.get_edges() if edge_order is None else list(edge_order)
    nodes = h.get_nodes()
    adj = {}

    for node in nodes:
        adj[node] = h.get_incident_edges(node)

    edge_to_id = {}
    id_to_edge = {}
    cont = 0
    for e in edges:
        edge_key = h._normalize_edge(e)
        edge_to_id[edge_key] = cont
        id_to_edge[cont] = edge_key
        cont += 1

    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(h))])

    vis = {}

    for n in adj:
        for i in range(len(adj[n]) - 1):
            for j in range(i + 1, len(adj[n])):
                id_i = edge_to_id[adj[n][i]]
                id_j = edge_to_id[adj[n][j]]
                k = frozenset((id_i, id_j))
                e_i = set(adj[n][i])
                e_j = set(adj[n][j])
                if k not in vis:
                    w = _distance(e_i, e_j)
                    if w >= s:
                        if weighted:
                            g.add_edge(id_i, id_j, weight=w)
                        else:
                            g.add_edge(id_i, id_j, weight=1)
                    vis[k] = True
    return g, id_to_edge


def directed_line_graph(
    h: DirectedHypergraph,
    distance="intersection",
    s=1,
    weighted=False,
    *,
    edge_order=None,
):
    """
    Returns a line graph of the directed hypergraph.

    Parameters
    ----------
    h : DirectedHypergraph
        The directed hypergraph to be projected.
    distance : str
        The distance function to be used. Can be 'intersection' or 'jaccard'.
    s : float
        The threshold for the distance function.
    weighted : bool
        Whether the line graph should be weighted or not.
    edge_order : list, optional (keyword-only)
        Explicit edge iteration order to make the returned `id_to_edge` mapping deterministic.
        If None, uses `h.get_edges()` order.

    Returns
    -------
    tuple
        `(g, id_to_edge)` where:
        - `g` is a `networkx.DiGraph` whose nodes are edge-ids `0..m-1`
        - `id_to_edge` maps those ids back to directed hyperedges

    Notes
    -----
    Computing the line graph can be very expensive for large hypergraphs.
    This function is deterministic given `edge_order`. Without it, edge-id assignment
    depends on the insertion/iteration order of `h`.

    Example
    -------
    >>> import networkx as nx
    >>> import hypergraphx as hgx
    >>> from hypergraphx.representations.projections import line_graph
    >>>
    >>> h = hgx.Hypergraph()
    >>> h.add_nodes([1, 2, 3, 4, 5])
    >>> h.add_edges([(1, 2), (1, 2, 3), (3, 4, 5)])
    >>> g, idx = line_graph(h)
    >>> g.edges()
    EdgeView([(0, 1), (1, 2)])
    """

    def _distance(a, b):
        if distance == "intersection":
            return intersection(a, b)
        if distance == "jaccard":
            return jaccard_similarity(a, b)

    edges = h.get_edges() if edge_order is None else list(edge_order)
    edge_to_id = {}
    id_to_edge = {}
    cont = 0
    for e in edges:
        edge_to_id[e] = cont
        id_to_edge[cont] = e
        cont += 1

    g = nx.DiGraph()
    g.add_nodes_from([i for i in range(len(h))])

    for edge1 in edges:
        for edge2 in edges:
            if edge1 != edge2:
                source = set(edge1[1])
                target = set(edge2[0])
                w = _distance(source, target)
                if w >= s:
                    if weighted:
                        g.add_edge(edge_to_id[edge1], edge_to_id[edge2], weight=w)
                    else:
                        g.add_edge(edge_to_id[edge1], edge_to_id[edge2])

    return g, id_to_edge
