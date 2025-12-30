class SerializationMixin:
    """
    Serialization and hashing helpers for hypergraph-like classes.
    """

    def expose_data_structures(self):
        data = {
            "type": self._type_name(),
            "_weighted": self._weighted,
            "_edge_list": self._edge_list,
            "_weights": self._weights,
            "hypergraph_metadata": self._hypergraph_metadata,
            "node_metadata": self._node_metadata,
            "edge_metadata": self._edge_metadata,
            "incidences_metadata": self._incidences_metadata,
            "reverse_edge_list": self._reverse_edge_list,
            "next_edge_id": self._next_edge_id,
        }
        if hasattr(self, "_empty_edges"):
            data["empty_edges"] = self._empty_edges
        data.update(self._expose_adjacency_data())
        data.update(self._extra_data_structures())
        return data

    def populate_from_dict(self, data):
        self._weighted = data.get("_weighted", False)
        self._edge_list = data.get("_edge_list", {}) or {}
        self._weights = data.get("_weights", {}) or {}
        self._hypergraph_metadata = data.get("hypergraph_metadata", {}) or {}
        self._node_metadata = data.get("node_metadata", {}) or {}
        self._edge_metadata = data.get("edge_metadata", {}) or {}
        self._reverse_edge_list = data.get("reverse_edge_list", {}) or {}
        self._next_edge_id = data.get("next_edge_id", 0)
        self._incidences_metadata = data.get("incidences_metadata", {}) or {}
        if hasattr(self, "_empty_edges"):
            self._empty_edges = data.get("empty_edges", {})
        self._populate_adjacency_data(data)
        self._populate_extra_data(data)

    def expose_attributes_for_hashing(self):
        edges = []
        for edge_key in sorted(self._edge_list.keys()):
            edge_id = self._edge_list[edge_key]
            edges.append(
                {
                    "nodes": self._hash_edge_nodes(edge_key),
                    "weight": self._weights.get(edge_id, 1),
                    "metadata": self._edge_metadata.get(edge_id, {}),
                }
            )

        nodes = []
        for node in sorted(self._node_metadata.keys()):
            nodes.append({"node": node, "metadata": self._node_metadata[node]})

        return {
            "type": self._type_name(),
            "weighted": self._weighted,
            "hypergraph_metadata": self._hypergraph_metadata,
            "edges": edges,
            "nodes": nodes,
        }

    def get_mapping(self):
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        encoder.fit(self.get_nodes())
        return encoder
