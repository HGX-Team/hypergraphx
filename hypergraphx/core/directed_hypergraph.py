import copy
from typing import Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder


class DirectedHypergraph:
    """
    A directed hypergraph is a set of nodes and a set of hyperedges, where each hyperedge is a directed relation between
    two sets of nodes (source nodes and target nodes). Hyperedges are represented as a tuple of (source_nodes, target_nodes),
    and each hyperedge can optionally have a weight.

    Parameters
    ----------
    edge_list : list of tuples of tuples
        A list of tuples representing the directed hyperedges of the form (source_nodes, target_nodes).
    weighted : bool
        Whether the hypergraph is weighted.
    weights : list of floats, optional
        A list of weights for the hyperedges.
    """

    def __init__(self, edge_list=None, hypergraph_metadata=None, weighted=False, weights=None, edge_metadata=None):
        self._weighted = weighted
        self._adj_out = {}
        self._adj_in = {}
        self._edge_list = {}
        if hypergraph_metadata is None:
            self.hypergraph_metadata = {}
        else:
            self.hypergraph_metadata = hypergraph_metadata
        self.hypergraph_metadata['weighted'] = weighted
        self.node_metadata = {}
        self.edge_metadata = {}

        if edge_list is not None:
            self.add_edges(edge_list, weights=weights, metadata=edge_metadata)

    def set_node_metadata(self, node, metadata):
        if node not in self._adj_out:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self.node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._adj_out:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self.node_metadata[node]

    def get_all_nodes_metadata(self):
        return list(self.node_metadata.values())

    def set_edge_metadata(self, edge, metadata):
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self.edge_metadata[edge] = metadata

    def get_edge_metadata(self, edge):
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self.edge_metadata[edge]

    def get_mapping(self):
        """
        Map the nodes of the hypergraph to integers in [0, n_nodes).

        Returns
        -------
        LabelEncoder
            The mapping.
        """
        encoder = LabelEncoder()
        encoder.fit(self.get_nodes())
        return encoder


    def add_node(self, node, metadata=None):
        """
        Add a node to the hypergraph. If the node is already in the hypergraph, nothing happens.

        Parameters
        ----------
        node : object
            The node to add.

        Returns
        -------
        None
        """
        if node not in self._adj_out:
            self._adj_out[node] = set()
            self._adj_in[node] = set()
            if metadata is None:
                self.node_metadata[node] = {}
            else:
                self.node_metadata[node] = metadata

    def add_nodes(self, node_list: list):
        """
        Add a list of nodes to the hypergraph.

        Parameters
        ----------
        node_list : list
            The list of nodes to add.

        Returns
        -------
        None
        """
        for node in node_list:
            self.add_node(node)

    def is_weighted(self):
        """
        Check if the hypergraph is weighted.

        Returns
        -------
        bool
            True if the hypergraph is weighted, False otherwise.
        """
        return self._weighted

    def get_sizes(self):
        """Returns the list of sizes of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of sizes of the hyperedges in the hypergraph.

        """
        return [len(edge[0]) + len(edge[1]) for edge in self._edge_list.keys()]

    def get_orders(self):
        """Returns the list of orders of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of orders of the hyperedges in the hypergraph.

        """
        return [len(edge[0]) + len(edge[1]) - 1 for edge in self._edge_list.keys()]

    def get_sources(self):
        """Returns the list of sources of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of sources of the hyperedges in the hypergraph.

        """
        return [edge[0] for edge in self._edge_list.keys()]

    def get_targets(self):
        """Returns the list of targets of the hyperedges in the hypergraph.

        Returns
        -------
        list
            List of targets of the hyperedges in the hypergraph.

        """
        return [edge[1] for edge in self._edge_list.keys()]

    def add_edge(self, edge: Tuple[Tuple, Tuple], weight=None, metadata=None):
        """Add a directed hyperedge to the hypergraph. If the hyperedge already exists, its weight is updated.

        Parameters
        ----------
        edge : tuple of tuples
            The directed hyperedge to add, represented as (source_nodes, target_nodes).
        weight : float, optional
            The weight of the hyperedge. If the hypergraph is weighted, this must be provided.
        metadata : dict, optional
            The metadata of the hyperedge.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the hypergraph is weighted and no weight is provided or if the hypergraph is not weighted and a weight is provided.
        """
        source, target = edge
        source = tuple(sorted(source))
        target = tuple(sorted(target))
        edge = (source, target)

        if self._weighted and weight is None:
            raise ValueError("If the hypergraph is weighted, a weight must be provided.")
        if not self._weighted and weight is not None:
            raise ValueError("If the hypergraph is not weighted, no weight must be provided.")

        if weight is None:
            if edge in self._edge_list and self._weighted:
                self._edge_list[edge] += 1
            else:
                self._edge_list[edge] = 1
        else:
            self._edge_list[edge] = weight

        for node in source:
            self.add_node(node)
            self._adj_out[node].add(edge)

        for node in target:
            self.add_node(node)
            self._adj_in[node].add(edge)

        if metadata is not None:
            self.set_edge_metadata(edge, metadata)
        else:
            self.set_edge_metadata(edge, {})

    def add_edges(self, edge_list: List[Tuple[Tuple, Tuple]], weights=None, metadata=None):
        """Add a list of directed hyperedges to the hypergraph. If a hyperedge is already in the hypergraph, its weight is updated.

        Parameters
        ----------
        edge_list : list of tuples of tuples
            The list of directed hyperedges to add (each as (source_nodes, target_nodes)).
        weights : list, optional
            The list of weights of the hyperedges.
        metadata : list, optional
            The list of metadata of the hyperedges.

        Returns
        -------
        None
        """
        if weights is not None and not self._weighted:
            print("Warning: weights are provided but the hypergraph is not weighted. The hypergraph will be weighted.")
            self._weighted = True

        if self._weighted and weights is not None:
            if len(edge_list) != len(weights):
                raise ValueError("The number of edges and weights must be the same.")

        for i, edge in enumerate(edge_list):
            self.add_edge(edge, weight=weights[i] if weights else None,
                          metadata=metadata[i] if metadata else None)

    def get_incident_in_edges(self, node, order=None, size=None):
        """
        Get the incident in-edges of a node.

        Parameters
        ----------
        node
        order
        size

        Returns
        -------
        list
            The list of incident in-edges.
        """
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return list(self._adj_in[node])
        elif size is not None:
            return [edge for edge in self._adj_in[node] if self._get_edge_size(edge) == size]
        elif order is not None:
            return [edge for edge in self._adj_in[node] if self._get_edge_size(edge) - 1 == order]

    def get_incident_out_edges(self, node, order=None, size=None):
        """
        Get the incident out-edges of a node.

        Parameters
        ----------
        node
        order
        size

        Returns
        -------
        list
            The list of incident out-edges.
        """
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return list(self._adj_out[node])
        elif size is not None:
            return [edge for edge in self._adj_out[node] if self._get_edge_size(edge) == size]
        elif order is not None:
            return [edge for edge in self._adj_out[node] if self._get_edge_size(edge) - 1 == order]

    def _get_edge_size(self, edge):
        """
        Get the size of a hyperedge.

        Parameters
        ----------
        edge

        Returns
        -------
        int
            The size of the hyperedge.
        """
        return len(edge[0]) + len(edge[1])

    def get_edges(
            self,
            order=None,
            size=None,
            up_to=False,
            subhypergraph=False,
            keep_isolated_nodes=False,
            metadata=False
    ):
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if not subhypergraph and keep_isolated_nodes:
            raise ValueError("Cannot keep nodes if not returning subhypergraphs.")

        if order is None and size is None:
            edges = list(self._edge_list.keys())
        else:
            if size is not None:
                order = size - 1
            if not up_to:
                edges = [edge for edge in list(self._edge_list.keys()) if self._get_edge_size(edge) - 1 == order]
            else:
                edges = [edge for edge in list(self._edge_list.keys()) if self._get_edge_size(edge) - 1 <= order]

        if subhypergraph and keep_isolated_nodes:
            h = DirectedHypergraph(weighted=self._weighted)
            h.add_nodes(list(self.get_nodes()))
            if self._weighted:
                edge_weights = [self.get_weight(edge) for edge in edges]
                h.add_edges(edges, edge_weights)
            else:
                h.add_edges(edges)

            for node in h.get_nodes():
                h.set_node_metadata(node, self.get_node_metadata(node))
            for edge in edges:
                h.set_edge_metadata(edge, self.get_edge_metadata(edge))
            return h

        elif subhypergraph:
            h = DirectedHypergraph(weighted=self._weighted)
            if self._weighted:
                edge_weights = [self.get_weight(edge) for edge in edges]
                h.add_edges(edges, edge_weights)
            else:
                h.add_edges(edges)

            for edge in edges:
                h.set_edge_metadata(edge, self.get_edge_metadata(edge))
            return h
        else:
            return edges if not metadata else {edge: self.get_edge_metadata(edge) for edge in edges}


    def remove_edge(self, edge: Tuple[Tuple, Tuple]):
        """Remove a directed edge from the hypergraph.

        Parameters
        ----------
        edge : tuple of tuples
            The edge to remove (source_nodes, target_nodes).

        Returns
        -------
        None
        """
        try:
            del self._edge_list[edge]
            source, target = edge

            # Remove from adjacency
            for node in source:
                self._adj_out[node].remove(edge)

            for node in target:
                self._adj_in[node].remove(edge)

        except KeyError:
            print(f"Edge {edge} not in hypergraph.")

    def remove_node(self, node, keep_edges=False):
        """Remove a node from the hypergraph, with an option to keep or remove edges incident to it."""
        if node not in self._adj_out or node not in self._adj_in:
            raise KeyError(f"Node {node} not in hypergraph.")

        # Handle incident edges
        if not keep_edges:
            outgoing_edges = list(self._adj_out[node])
            incoming_edges = list(self._adj_in[node])
            for edge in outgoing_edges + incoming_edges:
                self.remove_edge(edge)

        del self._adj_out[node]
        del self._adj_in[node]

    def get_nodes(self, metadata=False):
        """Returns the list of nodes in the hypergraph."""
        if not metadata:
            list(self._adj_out.keys())
        else:
            return {node: self.node_metadata[node] for node in self._adj_out.keys()}

    def num_nodes(self):
        """Returns the number of nodes in the hypergraph."""
        return len(self.get_nodes())

    def num_edges(self):
        """Returns the number of directed edges in the hypergraph."""
        return len(self._edge_list)

    def get_weight(self, edge: Tuple[Tuple, Tuple]):
        """Returns the weight of the specified directed edge."""
        return self._edge_list.get(edge)

    def set_weight(self, edge: Tuple[Tuple, Tuple], weight: float):
        """Sets the weight of the specified directed edge."""
        if edge in self._edge_list:
            self._edge_list[edge] = weight
        else:
            raise ValueError(f"Edge {edge} not in hypergraph.")

    def check_node(self, node):
        """Checks if the specified node is in the hypergraph.

        Parameters
        ----------
        node : Object
            The node to check.

        Returns
        -------
        bool
            True if the node is in the hypergraph, False otherwise.

        """
        return node in self._adj_out

    def clear(self):
        self._edge_list.clear()
        self._adj_out.clear()
        self._adj_in.clear()

    def check_edge(self, edge: Tuple[Tuple, Tuple]):
        """Checks if the specified edge is in the hypergraph.

        Parameters
        ----------
        edge : tuple
            The edge to check.

        Returns
        -------
        bool
            True if the edge is in the hypergraph, False otherwise.

        """
        return edge in self._edge_list

    def get_hypergraph_metadata(self):
        return self.hypergraph_metadata

    def set_hypergraph_metadata(self, metadata):
        self.hypergraph_metadata = metadata
