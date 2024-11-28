from typing import Tuple, List

from sklearn.preprocessing import LabelEncoder


class DirectedHypergraph:
    """
    A Directed Hypergraph is a generalization of a graph in which hyperedges have a direction.
    Each hyperedge connects a set of source nodes to a set of target nodes.
    """

    def __init__(
        self,
        edge_list=None,
        weighted=False,
        weights=None,
        hypergraph_metadata=None,
        node_metadata=None,
        edge_metadata=None,
    ):
        """
        Initialize a Directed Hypergraph.

        Parameters
        ----------
        edge_list : list of tuples of tuples, optional
            A list of directed hyperedges represented as (source_nodes, target_nodes),
            where source_nodes and target_nodes are tuples of nodes.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is False.
        weights : list of floats, optional
            A list of weights corresponding to the edges in `edge_list`. Required if `weighted` is True.
        hypergraph_metadata : dict, optional
            Metadata for the hypergraph. Default is an empty dictionary.
        node_metadata : dict, optional
            A dictionary of metadata for nodes, where keys are node identifiers and values are metadata dictionaries.
        edge_metadata : list of dicts, optional
            A list of metadata dictionaries corresponding to the edges in `edge_list`.

        Raises
        ------
        ValueError
            If `edge_list` and `weights` have mismatched lengths when `weighted` is True.
            If `edge_list` contains improperly formatted edges.
        """
        # Initialize hypergraph metadata
        self._hypergraph_metadata = hypergraph_metadata or {}
        self._hypergraph_metadata.update(
            {"weighted": weighted, "type": "DirectedHypergraph"}
        )

        # Initialize core attributes
        self._weighted = weighted
        self._adj_source = {}
        self._adj_target = {}
        self._edge_list = {}
        self._node_metadata = {}
        self._edge_metadata = {}
        self._reverse_edge_list = {}
        self._weights = {}
        self._next_edge_id = 0

        # Add node metadata if provided
        if node_metadata:
            for node, metadata in node_metadata.items():
                self.add_node(node, metadata=metadata)

        # Validate and add edges
        if edge_list is not None:
            if weighted and weights is not None and len(edge_list) != len(weights):
                raise ValueError("Edge list and weights must have the same length.")
            self.add_edges(edge_list, weights=weights, metadata=edge_metadata)

    def get_edge_list(self):
        return self._edge_list

    def set_edge_list(self, edge_list):
        self._edge_list = edge_list

    def get_adj_dict(self, source_target):
        if source_target == "source":
            return self._adj_source
        elif source_target == "target":
            return self._adj_target
        else:
            raise ValueError(
                "Invalid value for source_target. Must be 'source' or 'target'."
            )

    def set_adj_dict(self, adj_dict, source_target):
        if source_target == "source":
            self._adj_source = adj_dict
        elif source_target == "target":
            self._adj_target = adj_dict
        else:
            raise ValueError(
                "Invalid value for source_target. Must be 'source' or 'target'."
            )

    def set_node_metadata(self, node, metadata):
        if node not in self._adj_source:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node] = metadata

    def get_node_metadata(self, node):
        if node not in self._adj_source:
            raise ValueError("Node {} not in hypergraph.".format(node))
        return self._node_metadata[node]

    def get_all_nodes_metadata(self):
        return list(self._node_metadata.values())

    def set_edge_metadata(self, edge, metadata):
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        idx = self._edge_list[edge]
        self._edge_metadata[idx] = metadata

    def get_edge_metadata(self, edge):
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if edge not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        idx = self._edge_list[edge]
        return self._edge_metadata[idx]

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
        if metadata is None:
            self._node_metadata[node] = {}
        if node not in self._adj_source:
            self._adj_source[node] = []
            self._adj_target[node] = []
            self._node_metadata[node] = {}
        if self._node_metadata[node] == {}:
            self._node_metadata[node] = metadata

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

        if not self._weighted and weight is not None and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )

        if weight is None:
            weight = 1

        if edge not in self._edge_list:
            idx = self._next_edge_id
            self._next_edge_id += 1
            self._edge_list[edge] = idx
            self._reverse_edge_list[idx] = edge
            self._weights[idx] = 1 if not self._weighted else weight
        elif edge in self._edge_list and self._weighted:
            self._weights[self._edge_list[edge]] += weight

        idx = self._edge_list[edge]
        for node in source:
            self.add_node(node)
            self._adj_source[node].append(idx)

        for node in target:
            self.add_node(node)
            self._adj_target[node].append(idx)

        if metadata is not None:
            self.set_edge_metadata(edge, metadata)
        else:
            self.set_edge_metadata(edge, {})

    def add_edges(
        self, edge_list: List[Tuple[Tuple, Tuple]], weights=None, metadata=None
    ):
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
            print(
                "Warning: weights are provided but the hypergraph is not weighted. The hypergraph will be weighted."
            )
            self._weighted = True

        if self._weighted and weights is not None:
            if len(edge_list) != len(weights):
                raise ValueError("The number of edges and weights must be the same.")

        for i, edge in enumerate(edge_list):
            self.add_edge(
                edge,
                weight=weights[i] if weights else None,
                metadata=metadata[i] if metadata else None,
            )

    def get_source_edges(self, node, order=None, size=None):
        """
        Get the source edges in which a node is in the source set.

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
        if node not in self._adj_source:
            raise ValueError(f"Node {node} not in hypergraph.")
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return [self._reverse_edge_list[e_idx] for e_idx in self._adj_source[node]]
        elif size is not None:
            return [
                self._reverse_edge_list[e_idx]
                for e_idx in self._adj_source[node]
                if self._get_edge_size(self._reverse_edge_list[e_idx]) == size
            ]
        elif order is not None:
            return [
                self._reverse_edge_list[e_idx]
                for e_idx in self._adj_source[node]
                if self._get_edge_size(self._reverse_edge_list[e_idx]) - 1 == order
            ]

    def get_target_edges(self, node, order=None, size=None):
        """
        Get the hyperedges in which a node is in the target set.

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
        if node not in self._adj_target:
            raise ValueError(f"Node {node} not in hypergraph.")
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
        if order is None and size is None:
            return [self._reverse_edge_list[e_idx] for e_idx in self._adj_target[node]]
        elif size is not None:
            return [
                self._reverse_edge_list[e_idx]
                for e_idx in self._adj_target[node]
                if self._get_edge_size(self._reverse_edge_list[e_idx]) == size
            ]
        elif order is not None:
            return [
                self._reverse_edge_list[e_idx]
                for e_idx in self._adj_target[node]
                if self._get_edge_size(self._reverse_edge_list[e_idx]) - 1 == order
            ]

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
        metadata=False,
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
                edges = [
                    edge
                    for edge in list(self._edge_list.keys())
                    if self._get_edge_size(edge) - 1 == order
                ]
            else:
                edges = [
                    edge
                    for edge in list(self._edge_list.keys())
                    if self._get_edge_size(edge) - 1 <= order
                ]

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
            return (
                edges
                if not metadata
                else {edge: self.get_edge_metadata(edge) for edge in edges}
            )

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
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if edge in self._edge_list:
            e_idx = self._edge_list[edge]
            source, target = edge

            # Remove from adjacency
            for node in source:
                self._adj_source[node].remove(e_idx)

            for node in target:
                self._adj_target[node].remove(e_idx)

            del self._reverse_edge_list[e_idx]
            del self._weights[e_idx]
            del self._edge_metadata[e_idx]
            del self._edge_list[edge]

        else:
            raise ValueError(f"Edge {edge} not in hypergraph.")

    def remove_node(self, node, keep_edges=False):
        """Remove a node from the hypergraph, with an option to keep or remove edges incident to it."""
        if node not in self._adj_source or node not in self._adj_target:
            raise KeyError(f"Node {node} not in hypergraph.")

        # Handle incident edges
        if not keep_edges:
            target_edges = self.get_target_edges(node)
            source_edges = self.get_source_edges(node)
            for edge in source_edges:
                self.remove_edge(edge)
            for edge in target_edges:
                self.remove_edge(edge)

        del self._adj_source[node]
        del self._adj_target[node]

    def get_nodes(self, metadata=False):
        """Returns the list of nodes in the hypergraph."""
        if not metadata:
            return list(self._adj_source.keys())
        else:
            return {node: self._node_metadata[node] for node in self._adj_source.keys()}

    def num_nodes(self):
        """Returns the number of nodes in the hypergraph."""
        return len(self.get_nodes())

    def num_edges(self):
        """Returns the number of directed edges in the hypergraph."""
        return len(self._edge_list)

    def get_weight(self, edge: Tuple[Tuple, Tuple]):
        """Returns the weight of the specified directed edge."""
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if edge in self._edge_list:
            idx = self._edge_list[edge]
            return self._weights[idx]
        else:
            raise ValueError(f"Edge {edge} not in hypergraph.")

    def set_weight(self, edge: Tuple[Tuple, Tuple], weight: float):
        """Sets the weight of the specified directed edge."""
        if not self._weighted and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if edge in self._edge_list:
            idx = self._edge_list[edge]
            self._weights[idx] = weight
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
        return node in self._adj_source

    def clear(self):
        self._edge_list.clear()
        self._adj_source.clear()
        self._adj_target.clear()

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
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        return edge in self._edge_list

    def get_hypergraph_metadata(self):
        return self._hypergraph_metadata

    def set_hypergraph_metadata(self, metadata):
        self._hypergraph_metadata = metadata

    def set_attr_to_hypergraph_metadata(self, field, value):
        self._hypergraph_metadata[field] = value

    def set_attr_to_node_metadata(self, node, field, value):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        self._node_metadata[node][field] = value

    def set_attr_to_edge_metadata(self, edge, field, value):
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if edge not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._edge_metadata[self._edge_list[edge]][field] = value

    def remove_attr_from_node_metadata(self, node, field):
        if node not in self._node_metadata:
            raise ValueError("Node {} not in hypergraph.".format(node))
        del self._node_metadata[node][field]

    def remove_attr_from_edge_metadata(self, edge, field):
        edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
        if edge not in self._edge_metadata:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        del self._edge_metadata[self._edge_list[edge]][field]

    def expose_data_structures(self):
        """
        Expose the internal data structures of the directed hypergraph for serialization.

        Returns
        -------
        dict
            A dictionary containing all internal attributes of the directed hypergraph.
        """
        return {
            "type": "DirectedHypergraph",
            "_weighted": self._weighted,
            "_adj_source": self._adj_source,
            "_adj_target": self._adj_target,
            "_edge_list": self._edge_list,
            "_weights": self._weights,
            "hypergraph_metadata": self._hypergraph_metadata,
            "node_metadata": self._node_metadata,
            "edge_metadata": self._edge_metadata,
            "reverse_edge_list": self._reverse_edge_list,
            "next_edge_id": self._next_edge_id,
        }

    def populate_from_dict(self, data):
        """
        Populate the attributes of the directed hypergraph from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the attributes to populate the hypergraph.
        """
        self._weighted = data.get("_weighted", False)
        self._adj_source = data.get("_adj_source", {})
        self._adj_target = data.get("_adj_target", {})
        self._edge_list = data.get("_edge_list", {})
        self._weights = data.get("_weights", {})
        self._hypergraph_metadata = data.get("hypergraph_metadata", {})
        self._node_metadata = data.get("node_metadata", {})
        self._edge_metadata = data.get("edge_metadata", {})
        self._reverse_edge_list = data.get("reverse_edge_list", {})
        self._next_edge_id = data.get("next_edge_id", 0)

    def expose_attributes_for_hashing(self):
        """
        Expose relevant attributes for hashing specific to DirectedHypergraph.

        Returns
        -------
        dict
            A dictionary containing key attributes.
        """
        edges = []
        for edge in sorted(self._edge_list.keys()):
            sorted_edge = (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
            edge_id = self._edge_list[edge]
            edges.append(
                {
                    "nodes": sorted_edge,
                    "weight": self._weights.get(edge_id, 1),
                    "metadata": self._edge_metadata.get(edge_id, {}),
                }
            )

        nodes = []
        for node in sorted(self._node_metadata.keys()):
            nodes.append({"node": node, "metadata": self._node_metadata[node]})

        return {
            "type": "DirectedHypergraph",
            "weighted": self._weighted,
            "hypergraph_metadata": self._hypergraph_metadata,
            "edges": edges,
            "nodes": nodes,
        }
