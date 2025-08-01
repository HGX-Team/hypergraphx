import copy
import math
from typing import Tuple, Any, List, Dict, Optional, Union
from collections import Counter

from sklearn.preprocessing import LabelEncoder

from hypergraphx import Hypergraph
from hypergraphx.core.i_undirected_hypergraph import IUndirectedHypergraph


def _get_size(edge):
    """Get the size of an edge."""
    if len(edge) == 2 and isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
        return len(edge[0]) + len(edge[1])
    else:
        return len(edge)


def _get_order(edge):
    """Get the order of an edge."""
    return _get_size(edge) - 1


def _get_nodes(edge):
    """Get all nodes from an edge."""
    if len(edge) == 2 and isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
        return list(edge[0]) + list(edge[1])
    else:
        return list(edge)


class TemporalHypergraph(IUndirectedHypergraph):
    """
    A Temporal Hypergraph is a hypergraph where each hyperedge is associated with a specific timestamp.
    Temporal hypergraphs are useful for modeling systems where interactions between nodes change over time, such as social networks,
    communication networks, and transportation systems.
    """

    def __init__(
        self,
        edge_list: Optional[List]=None,
        time_list: Optional[List]=None,
        weighted: bool = False,
        weights: Optional[List[int]]=None,
        hypergraph_metadata: Optional[Dict] = None,
        node_metadata: Optional[Dict] = None,
        edge_metadata: Optional[List[Dict]] = None
    ):
        """
        Initialize a Temporal Hypergraph with optional edges, times, weights, and metadata.

        Parameters
        ----------
        edge_list : list of tuples, optional
            A list of edges where each edge is represented as a tuple of nodes.
            If `time_list` is not provided, each tuple in `edge_list` should
            have the format `(time, edge)`, where `edge` is itself a tuple of nodes.
        time_list : list of int, optional
            A list of times corresponding to each edge in `edge_list`.
            Must be provided if `edge_list` does not include time information.
        weighted : bool, optional
            Indicates whether the hypergraph is weighted. Default is False.
        weights : list of float, optional
            A list of weights for each edge in `edge_list`. Must be provided if `weighted` is True.
        hypergraph_metadata : dict, optional
            Metadata for the hypergraph as a whole. Default is an empty dictionary.
        node_metadata : dict, optional
            A dictionary of metadata for nodes, where keys are node identifiers and values are metadata dictionaries.
        edge_metadata : list of dict, optional
            A list of metadata dictionaries for each edge in `edge_list`.

        Raises
        ------
        ValueError
            If `edge_list` and `time_list` have mismatched lengths.
            If `edge_list` contains improperly formatted edges when `time_list` is None.
            If `time_list` is provided without `edge_list`.
        """
        # Initialize base class with temporal-specific metadata
        temporal_metadata = hypergraph_metadata or {}
        temporal_metadata.update({"type": "TemporalHypergraph"})
        
        super().__init__(
            edge_list=None,  # We'll handle edge addition separately
            weighted=weighted,
            weights=None,
            hypergraph_metadata=temporal_metadata,
            node_metadata=node_metadata,
            edge_metadata=None  # We'll handle this separately
        )

        # Handle edge and time list consistency
        if edge_list is not None and time_list is None:
            # Extract times from the edge list if time information is embedded
            if not all(
                isinstance(edge, tuple) and len(edge) == 2 for edge in edge_list
            ):
                raise ValueError(
                    "If time_list is not provided, edge_list must contain tuples of the form (time, edge)."
                )
            time_list = [edge[0] for edge in edge_list]
            edge_list = [edge[1] for edge in edge_list]

        if edge_list is None and time_list is not None:
            raise ValueError("Edge list must be provided if time list is provided.")

        if edge_list is not None and time_list is not None:
            if len(edge_list) != len(time_list):
                raise ValueError("Edge list and time list must have the same length.")
            self.add_edges(
                edge_list,
                time_list,
                weights=weights,
                metadata=edge_metadata
            )

    # =============================================================================
    # Implementation of abstract methods from IUndirectedHypergraph
    # =============================================================================

    def _add_edge_implementation(self, edge, weight, metadata, time=None, **kwargs):
        """Implementation of abstract method for adding a single edge."""
        if time is None:
            raise ValueError("Time must be provided for temporal hypergraph edges.")
        
        if not isinstance(time, int):
            raise TypeError("Time must be an integer")

        if not self._weighted and weight is not None and weight != 1:
            raise ValueError(
                "If the hypergraph is not weighted, weight can be 1 or None."
            )
        if weight is None:
            weight = 1

        if time < 0:
            raise ValueError("Time must be a positive integer")

        _edge = self._canon_edge(edge)
        temporal_edge = (time, _edge)

        if temporal_edge not in self._edge_list:
            e_id = self._next_edge_id
            self._reverse_edge_list[e_id] = temporal_edge
            self._edge_list[temporal_edge] = e_id
            self._next_edge_id += 1
            self._weights[e_id] = weight
        elif temporal_edge in self._edge_list and self._weighted:
            self._weights[self._edge_list[temporal_edge]] += weight

        e_id = self._edge_list[temporal_edge]

        if metadata is None:
            metadata = {}
        self._edge_metadata[temporal_edge] = metadata

        nodes = _get_nodes(_edge)
        for node in nodes:
            self.add_node(node)

        for node in nodes:
            if e_id not in self._adj[node]:
                self._adj[node].append(e_id)

    def _extract_nodes_from_edge(self, edge) -> List:
        """Extract node list from a temporal edge representation."""
        if isinstance(edge, tuple) and len(edge) == 2:
            # Temporal edge format: (time, edge_nodes)
            time, edge_nodes = edge
            return _get_nodes(edge_nodes)
        else:
            # Fallback for direct edge nodes
            return _get_nodes(edge)

    def _get_edge_size(self, edge_key) -> int:
        """Get the size of an edge given its temporal key representation."""
        if isinstance(edge_key, tuple) and len(edge_key) == 2:
            # Temporal edge format: (time, edge_nodes)
            time, edge_nodes = edge_key
            return _get_size(edge_nodes)
        else:
            return _get_size(edge_key)

    def remove_node(self, node: Any, keep_edges: bool = False) -> None:
        """Remove a node from the temporal hypergraph."""
        if node not in self._adj:
            raise ValueError(f"Node {node} not in hypergraph.")

        edges_to_process = list(self._adj[node])

        if keep_edges:
            for edge_id in edges_to_process:
                time, edge = self._reverse_edge_list[edge_id]
                updated_edge = tuple(n for n in edge if n != node)

                self.remove_edge(edge, time)
                if updated_edge:
                    self.add_edge(
                        updated_edge,
                        time,
                        weight=self._weights.get(edge_id, 1),
                        metadata=self.get_edge_metadata(edge=edge, time=time),
                    )
        else:
            for edge_id in edges_to_process:
                time, edge = self._reverse_edge_list[edge_id]
                self.remove_edge(edge, time)

        del self._adj[node]
        if node in self._node_metadata:
            del self._node_metadata[node]

    def add_edge(self, edge, time: int, weight=None, metadata=None) -> None:
        """
        Add an edge to the temporal hypergraph. If the edge already exists, the weight is updated.

        Parameters
        ----------
        edge : tuple
            The edge to add.
        time: int
            The time at which the edge occurs.
        weight: float, optional
            The weight of the edge. Default is None.
        metadata: dict, optional
            Metadata for the edge. Default is an empty dictionary.

        Raises
        ------
        TypeError
            If time is not an integer.
        ValueError
            If the hypergraph is not weighted and weight is not None or 1.
        """
        self._add_edge_implementation(edge, weight, metadata, time=time)

    def remove_edge(self, edge, time: int) -> None:
        """Remove an edge from the temporal hypergraph."""
        _edge = self._canon_edge(edge)
        temporal_edge = (time, _edge)
        
        if temporal_edge not in self._edge_list:
            raise ValueError(f"Edge {temporal_edge} not in hypergraph.")
        
        edge_id = self._edge_list[temporal_edge]

        # Remove edge from reverse lookup and metadata
        del self._reverse_edge_list[edge_id]
        if edge_id in self._weights:
            del self._weights[edge_id]
        if temporal_edge in self._edge_metadata.keys():
            del self._edge_metadata[temporal_edge]

        # Remove from adjacency lists
        nodes = _get_nodes(_edge)
        for node in nodes:
            if edge_id in self._adj[node]:
                self._adj[node].remove(edge_id)

        del self._edge_list[temporal_edge]

    def get_edges(
        self,
        time_window=None,
        order=None,
        size=None,
        up_to=False,
        metadata=False,
    ):
        """Get the edges in the temporal hypergraph."""
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")

        edges = []
        if time_window is None:
            edges = list(self._edge_list.keys())
        elif isinstance(time_window, tuple) and len(time_window) == 2:
            for _t, _edge in sorted(self._edge_list.keys()):
                if time_window[0] <= _t < time_window[1]:
                    edges.append((_t, _edge))
        else:
            raise ValueError("Time window must be a tuple of length 2 or None")
            
        if order is not None or size is not None:
            if size is not None:
                order = size - 1
            if not up_to:
                edges = [edge for edge in edges if len(edge[1]) - 1 == order]
            else:
                edges = [edge for edge in edges if len(edge[1]) - 1 <= order]
                
        return (
            edges
            if not metadata
            else {edge: self.get_edge_metadata(edge=edge[1], time=edge[0]) for edge in edges}
        )

    def get_incident_edges(self, node, order: int = None, size: int = None) -> List[Tuple]:
        """Get the incident hyperedges of a node in the hypergraph."""
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
            
        if order is None and size is None:
            return [self._reverse_edge_list[edge_id] for edge_id in self._adj[node]]
        else:
            if order is None:
                order = size - 1
            return [
                self._reverse_edge_list[edge_id]
                for edge_id in self._adj[node]
                if len(self._reverse_edge_list[edge_id][1]) - 1 == order
            ]

    # =============================================================================
    # Temporal-specific edge management methods
    # =============================================================================

    def add_edges(self, edge_list, time_list, weights=None, metadata=None) -> None:
        """Add multiple edges to the temporal hypergraph."""
        if not isinstance(edge_list, list) or not isinstance(time_list, list):
            raise TypeError("Edge list and time list must be lists")

        if len(edge_list) != len(time_list):
            raise ValueError("Edge list and time list must have the same length")

        if weights is not None and not self._weighted:
            print(
                "Warning: weights are provided but the hypergraph is not weighted. The hypergraph will be weighted."
            )
            self._weighted = True

        if self._weighted and weights is not None:
            if len(set(edge_list)) != len(list(edge_list)):
                raise ValueError(
                    "If weights are provided, the edge list must not contain repeated edges."
                )
            if len(list(edge_list)) != len(list(weights)):
                raise ValueError("The number of edges and weights must be the same.")

        for i, edge in enumerate(edge_list):
            self.add_edge(
                edge,
                time_list[i],
                weight=(
                    weights[i] if self._weighted and weights is not None else None
                ),
                metadata=metadata[i] if metadata is not None else None,
            )

    def remove_edges(self, edge_list) -> None:
        """Remove a list of edges from the hypergraph."""
        for edge in edge_list:
            if isinstance(edge, tuple) and len(edge) == 2:
                time, edge_nodes = edge
                self.remove_edge(edge_nodes, time)
            else:
                raise ValueError("Edge must be a tuple of (time, edge_nodes)")

    # =============================================================================
    # Override base class methods for temporal-specific behavior
    # =============================================================================

    def get_neighbors(self, node, order: int = None, size: int = None):
        """Get the neighbors of a node in the hypergraph."""
        if node not in self._adj:
            raise ValueError("Node {} not in hypergraph.".format(node))
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
            
        if order is None and size is None:
            neigh = set()
            edges = self.get_incident_edges(node)
            for edge in edges:
                neigh.update(_get_nodes(edge[1]))
            neigh.discard(node)
            return neigh
        else:
            if order is None:
                order = size - 1
            neigh = set()
            edges = self.get_incident_edges(node, order=order)
            for edge in edges:
                neigh.update(_get_nodes(edge[1]))
            neigh.discard(node)
            return neigh

    def num_edges(self, order: int = None, size: int = None, up_to: bool = False) -> int:
        """Get the number of edges in the hypergraph."""
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")

        if order is None and size is None:
            return len(self._edge_list)
        else:
            if size is not None:
                order = size - 1
            count = 0
            for edge_key in self._edge_list:
                edge_size = self._get_edge_size(edge_key)
                edge_order = edge_size - 1
                if not up_to:
                    if edge_order == order:
                        count += 1
                else:
                    if edge_order <= order:
                        count += 1
            return count

    def get_sizes(self) -> List[int]:
        """Get the size of each edge in the hypergraph."""
        return [_get_size(edge[1]) for edge in self._edge_list.keys()]

    def is_uniform(self) -> bool:
        """Check if the hypergraph is uniform."""
        if not self._edge_list:
            return True
            
        sizes = self.get_sizes()
        return len(set(sizes)) <= 1

    def get_weights(self, order=None, size=None, up_to=False, asdict=False):
        """Get weights of edges in the hypergraph."""
        w = None
        if order is not None and size is not None:
            raise ValueError("Order and size cannot be both specified.")
            
        if order is None and size is None:
            w = {
                edge: self._weights[self._edge_list[edge]] for edge in self.get_edges()
            }

        if size is not None:
            order = size - 1

        if w is None:
            w = {
                edge: self._weights[self._edge_list[edge]]
                for edge in self.get_edges(order=order, up_to=up_to)
            }

        if asdict:
            return w
        else:
            return list(w.values())

    # =============================================================================
    # Weight Management (Use base class with temporal edge format)
    # =============================================================================
    
    def get_weight(self, edge, time: int):
        """Get the weight of an edge at a specific time."""
        return super().get_weight(edge, time)

    def set_weight(self, edge, time: int, weight) -> None:
        """Set the weight of an edge at a specific time."""
        super().set_weight(edge, weight, time)

    # =============================================================================
    # Temporal-specific methods
    # =============================================================================

    def get_times_for_edge(self, edge):
        """Get the times at which a specific set of nodes forms a hyperedge."""
        edge = self._canon_edge(edge)
        times = []
        for time, _edge in self._edge_list.keys():
            if _edge == edge:
                times.append(time)
        return times

    def min_time(self):
        """Get the minimum time in the hypergraph."""
        if not self._edge_list:
            return None
        return min(edge[0] for edge in self._edge_list.keys())

    def max_time(self):
        """Get the maximum time in the hypergraph."""
        if not self._edge_list:
            return None
        return max(edge[0] for edge in self._edge_list.keys())

    def aggregate(self, time_window: int):
        """Aggregate edges within time windows."""
        if not isinstance(time_window, int) or time_window <= 0:
            raise TypeError("Time window must be a positive integer")

        aggregated = {}
        node_list = self.get_nodes()

        # Get all edges and determine the max time
        sorted_edges = sorted(self.get_edges())
        if not sorted_edges:
            return aggregated  # Return empty if no edges exist

        max_time = max(edge[0] for edge in sorted_edges)  # Maximum time of all edges

        # Initialize time window boundaries
        t_start = 0
        t_end = time_window
        edges_in_window = []
        num_windows_created = 0

        edge_index = 0  # Pointer to the current edge in sorted_edges

        while t_start <= max_time:
            # Collect edges for the current window
            while (
                edge_index < len(sorted_edges)
                and t_start <= sorted_edges[edge_index][0] < t_end
            ):
                edges_in_window.append(sorted_edges[edge_index])
                edge_index += 1

            # Create the hypergraph for this time window
            Hypergraph_t = Hypergraph(weighted=self._weighted)

            # Add edges to the hypergraph
            for time, edge_nodes in edges_in_window:
                Hypergraph_t.add_edge(
                    edge_nodes,
                    metadata=self.get_edge_metadata(edge=edge_nodes, time=time),
                    weight=self.get_weight(edge_nodes, time),
                )

            # Add all nodes to ensure node consistency
            for node in node_list:
                Hypergraph_t.add_node(node, metadata=self._node_metadata[node])

            # Store the finalized hypergraph for this window
            aggregated[num_windows_created] = Hypergraph_t
            num_windows_created += 1

            # Advance to the next time window
            t_start = t_end
            t_end += time_window
            edges_in_window = []  # Reset for the next window

        return aggregated

    def subhypergraph(
        self, time_window=None, add_all_nodes: bool = False
    ) -> dict[int, Hypergraph]:
        """Create a hypergraph for each time of the Temporal Hypergraph."""
        edges = self.get_edges()
        res = dict()
        if time_window is None:
            time_window = (-math.inf, math.inf)
        if not isinstance(time_window, tuple):
            raise ValueError("Time window must be a tuple of length 2 or None")

        for edge in edges:
            if time_window[0] <= edge[0] < time_window[1]:
                if edge[0] not in res.keys():
                    res[edge[0]] = Hypergraph(weighted=self.is_weighted())
                weight = self.get_weight(edge[1], edge[0])
                res[edge[0]].add_edge(edge[1], weight)
        if add_all_nodes:
            for node in self.get_nodes():
                for k, v in res.items():
                    if not v.check_node(node):
                        v.add_node(node)

        return res

    # =============================================================================
    # Metadata Management (Use base class with temporal edge format)
    # =============================================================================
    
    def get_incidence_metadata(self, edge, node, time: int = None):
        """Get incidence metadata for a specific edge-node pair."""
        edge = self._canon_edge(edge)
        k = (time, edge) if time is not None else edge
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        return self._incidences_metadata.get((k, node), {})

    def set_incidence_metadata(self, edge, node, metadata, time: int = None):
        """Set incidence metadata for a specific edge-node pair."""
        edge = self._canon_edge(edge)
        k = (time, edge) if time is not None else edge
        if k not in self._edge_list:
            raise ValueError("Edge {} not in hypergraph.".format(edge))
        self._incidences_metadata[(k, node)] = metadata

    def get_all_incidences_metadata(self):
        """Get all incidence metadata."""
        return {k: v for k, v in self._incidences_metadata.items()}

    def _restructure_query_edge(self, k: Tuple[Tuple, Any], time: int):
        """Helper for modifying a query edge prior to metadata retrieval."""
        return (time, k)

    # =============================================================================
    # Utility Methods (Override temporal-specific canonical edge handling)
    # =============================================================================
    
    def _canon_edge(self, edge: Tuple) -> Tuple:
        """Canonical form of an edge - sorts inner tuples."""
        edge = tuple(edge)

        if len(edge) == 2:
            if isinstance(edge[0], tuple) and isinstance(edge[1], tuple):
                # Sort the inner tuples and return
                return (tuple(sorted(edge[0])), tuple(sorted(edge[1])))
            elif not isinstance(edge[0], tuple) and not isinstance(edge[1], tuple):
                # Sort the edge itself if it contains IDs (non-tuple elements)
                return tuple(sorted(edge))

        return tuple(sorted(edge))

    # =============================================================================
    # Matrix operations (inherited from base class)
    # =============================================================================

    def temporal_adjacency_matrix(self, return_mapping: bool = False):
        """Get the temporal adjacency matrix."""
        from hypergraphx.linalg import temporal_adjacency_matrix
        return temporal_adjacency_matrix(self, return_mapping)

    def annealed_adjacency_matrix(self, return_mapping: bool = False):
        """Get the annealed adjacency matrix."""
        from hypergraphx.linalg import annealed_adjacency_matrix
        return annealed_adjacency_matrix(self, return_mapping)

    # =============================================================================
    # Serialization Support (Override base class methods)
    # =============================================================================
    
    def expose_data_structures(self) -> Dict:
        """Expose the internal data structures for serialization."""
        base_data = super().expose_data_structures()
        base_data["type"] = "TemporalHypergraph"
        return base_data

    def expose_attributes_for_hashing(self) -> dict:
        """Expose relevant attributes for hashing."""
        edges = []
        for edge in sorted(self._edge_list.keys()):
            edge = (edge[0], tuple(sorted(edge[1])))
            edge_id = self._edge_list[edge]
            edges.append(
                {
                    "nodes": edge,
                    "weight": self._weights.get(edge_id, 1),
                    "metadata": self.get_edge_metadata(edge=edge[1], time=edge[0]),
                }
            )

        nodes = []
        for node in sorted(self._node_metadata.keys()):
            nodes.append({"node": node, "metadata": self._node_metadata[node]})

        return {
            "type": "TemporalHypergraph",
            "weighted": self._weighted,
            "hypergraph_metadata": self._hypergraph_metadata,
            "edges": edges,
            "nodes": nodes,
        }