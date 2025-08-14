import pytest
import copy
import networkx as nx
from unittest.mock import Mock, patch

# Import the classes to test
from hypergraphx.core.Hypergraph import Hypergraph
from hypergraphx.viz.HypergraphVisualizer import HypergraphVisualizer


class TestHypergraphVisualizer:
    """Test suite for HypergraphVisualizer class."""

    @pytest.fixture
    def simple_hypergraph(self):
        """Create a simple hypergraph for testing."""
        h = Hypergraph()
        h.add_nodes([1, 2, 3, 4])
        h.add_edges([
            (1, 2),  # Order 1 edge
            (2, 3),  # Order 1 edge
            (1, 2, 3),  # Order 2 edge (hyperedge)
            (2, 3, 4),  # Order 2 edge (hyperedge)
        ])
        return h

    @pytest.fixture
    def hypergraph_with_metadata(self):
        """Create a hypergraph with metadata for testing."""
        h = Hypergraph()
        h.add_nodes([1, 2, 3, 4])
        
        # Add node metadata
        h.set_node_metadata(1, {"text": "Node1"})
        h.set_node_metadata(2, {"text": "Node2"})
        h.set_node_metadata(3, {"text": "Node3"})
        h.set_node_metadata(4, {"text": "Node4"})
        
        # Add edges with metadata
        h.add_edge((1, 2), metadata={"type": "edge_type_1"})
        h.add_edge((2, 3), metadata={"type": "edge_type_2"})
        h.add_edge((1, 2, 3), metadata={"type": "hyperedge_type_1"})
        h.add_edge((2, 3, 4), metadata={"type": "hyperedge_type_2"})
        
        return h

    @pytest.fixture
    def empty_hypergraph(self):
        """Create an empty hypergraph for testing."""
        return Hypergraph()

    @pytest.fixture
    def visualizer(self, simple_hypergraph):
        """Create a HypergraphVisualizer instance."""
        return HypergraphVisualizer(simple_hypergraph)

    @pytest.fixture
    def visualizer_with_metadata(self, hypergraph_with_metadata):
        """Create a HypergraphVisualizer instance with metadata."""
        return HypergraphVisualizer(hypergraph_with_metadata)

    def test_init(self, simple_hypergraph):
        """Test HypergraphVisualizer initialization."""
        visualizer = HypergraphVisualizer(simple_hypergraph)
        
        assert visualizer.g == simple_hypergraph
        assert visualizer.directed == False
        assert hasattr(visualizer, 'node_labels')
        assert hasattr(visualizer, 'pairwise_edge_labels')
        assert hasattr(visualizer, 'hyperedge_labels')

    def test_to_nx_returns_undirected_graph(self, visualizer):
        """Test that to_nx returns an undirected NetworkX graph."""
        nx_graph = visualizer.to_nx()
        
        assert isinstance(nx_graph, (nx.Graph, nx.DiGraph))
        # Since directed=False, it should return the pairwise subgraph
        assert hasattr(nx_graph, 'nodes')
        assert hasattr(nx_graph, 'edges')

    @patch.object(HypergraphVisualizer, 'get_pairwise_subgraph')
    def test_to_nx_calls_get_pairwise_subgraph(self, mock_get_pairwise, visualizer):
        """Test that to_nx calls get_pairwise_subgraph."""
        mock_graph = Mock()
        mock_get_pairwise.return_value = mock_graph
        
        result = visualizer.to_nx()
        
        mock_get_pairwise.assert_called_once()
        assert result == mock_graph

    def test_get_hyperedge_labels_with_metadata(self, visualizer_with_metadata):
        """Test getting hyperedge labels from metadata."""
        labels = visualizer_with_metadata.get_hyperedge_labels("type")
        
        # Should only include hyperedges (order > 1, i.e., size > 2)
        expected_hyperedges = {(1, 2, 3), (2, 3, 4)}
        
        for edge in labels.keys():
            assert len(edge) > 2  # Only hyperedges
            assert edge in expected_hyperedges

    def test_get_hyperedge_labels_empty_for_no_metadata(self, visualizer):
        """Test that get_hyperedge_labels returns empty dict when no metadata."""
        labels = visualizer.get_hyperedge_labels("type")
        
        # Should be empty since no metadata was added
        assert isinstance(labels, dict)
        assert len(labels) == 0

    def test_get_hyperedge_labels_custom_key(self, hypergraph_with_metadata):
        """Test getting hyperedge labels with custom metadata key."""
        # Add hyperedge with custom metadata key
        h = hypergraph_with_metadata
        h.add_edge((1, 3, 4), metadata={"custom_key": "custom_value"})
        
        visualizer = HypergraphVisualizer(h)
        labels = visualizer.get_hyperedge_labels("custom_key")
        
        # Should only contain the edge with the custom key
        assert (1, 3, 4) in labels
        assert labels[(1, 3, 4)] == "custom_value"

    def test_get_hyperedge_styling_data_structure(self, visualizer):
        """Test the structure of get_hyperedge_styling_data return value."""
        # Create a simple position dict
        pos = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
        hyperedge = (1, 2, 3)
        
        result = visualizer.get_hyperedge_styling_data(
            hyperedge, pos
        )
        
        # Should return tuple of (x_coords, y_coords, color, facecolor)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        x_coords, y_coords = result
        assert isinstance(x_coords, list)
        assert isinstance(y_coords, list)
        assert isinstance(visualizer.hyperedge_color_by_order[2], str)
        assert isinstance(visualizer.hyperedge_facecolor_by_order[2], str)
        
        # Colors should be hex strings
        assert visualizer.hyperedge_color_by_order[2].startswith('#')
        assert visualizer.hyperedge_facecolor_by_order[2].startswith('#')

    def test_get_hyperedge_styling_data_updates_color_dicts(self, visualizer):
        """Test that get_hyperedge_styling_data updates color dictionaries."""
        pos = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
        hyperedge = (1, 2, 3)  # Order 2 hyperedge
        
        visualizer.get_hyperedge_styling_data(
            hyperedge, pos
        )
        
        # Order of (1,2,3) is 2, so order 2 should be added to dicts
        assert 2 in visualizer.hyperedge_color_by_order
        assert 2 in visualizer.hyperedge_facecolor_by_order
        
        # Colors should be hex strings
        assert visualizer.hyperedge_color_by_order[2].startswith('#')
        assert visualizer.hyperedge_facecolor_by_order[2].startswith('#')

    def test_get_hyperedge_styling_data_different_sizes(self, visualizer):
        """Test get_hyperedge_styling_data with different hyperedge sizes."""
        # Test with triangle (3 nodes)
        pos_3 = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
        hyperedge_3 = (1, 2, 3)
        
        # Test with square (4 nodes)
        pos_4 = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}
        hyperedge_4 = (1, 2, 3, 4)
        
        # Both should work without errors
        visualizer1 = copy.deepcopy(visualizer)
        result_3 = visualizer1.get_hyperedge_styling_data(
            hyperedge_3, pos_3
        )
        
        visualizer2 = copy.deepcopy(visualizer)
        result_4 = visualizer2.get_hyperedge_styling_data(
            hyperedge_4, pos_4
        )
        
        assert len(result_3) == 2
        assert len(result_4) == 2
        
        # Different orders should have different colors
        assert 2 in visualizer1.hyperedge_color_by_order.keys()  # Order 2 (size 3)
        assert 3 in visualizer2.hyperedge_color_by_order.keys()  # Order 3 (size 4)

    def test_get_hyperedge_styling_data_maintains_existing_colors(self, visualizer):
        """Test that existing colors in the dictionaries are maintained."""
        pos = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
        hyperedge = (1, 2, 3)  # Order 2
        
        # Pre-populate the color dictionaries
        existing_color = "#123456"
        existing_facecolor = "#654321"
        
        visualizer.hyperedge_color_by_order.update({2: existing_color})
        visualizer.hyperedge_facecolor_by_order.update({2: existing_facecolor})

        result = visualizer.get_hyperedge_styling_data(
            hyperedge, pos
        )
        
        # Should use existing colors, not generate new ones
        assert visualizer.hyperedge_color_by_order[2] == existing_color
        assert visualizer.hyperedge_facecolor_by_order[2] == existing_facecolor

    @patch('hypergraphx.viz.HypergraphVisualizer.random.randint')
    def test_get_hyperedge_styling_data_random_color_generation(self, mock_randint, visualizer):
        """Test that random colors are generated correctly."""
        mock_randint.side_effect = [0x123456, 0x654321]  # Mock random color values
        
        pos = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
        hyperedge = (1, 2, 3)
        
        result = visualizer.get_hyperedge_styling_data(
            hyperedge, pos
        )

        # Should generate colors based on mocked random values
        assert visualizer.hyperedge_color_by_order[2] == "#123456"
        assert visualizer.hyperedge_facecolor_by_order[2] == "#654321"
        
        # Should call randint twice (once for color, once for facecolor)
        assert mock_randint.call_count == 2

    def test_inheritance_from_ihypergraph_visualizer(self, visualizer):
        """Test that HypergraphVisualizer properly inherits from IHypergraphVisualizer."""
        from hypergraphx.viz.IHypergraphVisualizer import IHypergraphVisualizer
        
        assert isinstance(visualizer, IHypergraphVisualizer)
        
        # Should have inherited attributes
        assert hasattr(visualizer, 'g')
        assert hasattr(visualizer, 'directed')
        assert hasattr(visualizer, 'node_labels')
        assert hasattr(visualizer, 'pairwise_edge_labels')
        assert hasattr(visualizer, 'hyperedge_labels')

    def test_directed_attribute_set_correctly(self, visualizer):
        """Test that the directed attribute is set to False for undirected hypergraphs."""
        assert visualizer.directed == False

    def test_empty_hypergraph_handling(self, empty_hypergraph):
        """Test that visualizer handles empty hypergraphs correctly."""
        visualizer = HypergraphVisualizer(empty_hypergraph)
        
        assert visualizer.g == empty_hypergraph
        assert visualizer.directed == False
        
        # Should not raise errors when getting labels from empty hypergraph
        labels = visualizer.get_hyperedge_labels("type")
        assert isinstance(labels, dict)
        assert len(labels) == 0
    
    @patch('hypergraphx.viz.Object')
    @patch('hypergraphx.viz.IHypergraphVisualizer.IHypergraphVisualizer.get_hyperedge_center_of_mass')
    def test_get_hyperedge_styling_data_uses_smoothing(self, mock_get_center, mock_object_class, visualizer):
        """Test that get_hyperedge_styling_data uses the Object smoothing functionality."""
        
        # Mock get_hyperedge_center_of_mass method from the base class
        mock_points = [(0, 0), (1, 0), (0.5, 1)]
        mock_get_center.return_value = (mock_points, 0.5, 0.33)
        
        # Set up Object mock
        mock_object = Mock()
        mock_object_class.return_value = mock_object
        
        # Initialize color dictionaries to avoid KeyError
        visualizer.hyperedge_color_by_order = {}
        visualizer.hyperedge_facecolor_by_order = {}
        
        pos = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
        hyperedge = (1, 2, 3)
        
        result = visualizer.get_hyperedge_styling_data(hyperedge, pos, number_of_refinements=4)

        # Verify Object was created
        mock_object_class.assert_called_once()
        
        # Verify smoothing was called
        mock_object.Smooth_by_Chaikin.assert_called_once_with(number_of_refinements=4)

        # Verify results
        x_coords, y_coords = result
        assert len(x_coords) == 49
        assert len(y_coords) == 49

    def test_get_hyperedge_styling_data_invalid_position(self, visualizer):
        """Test handling of invalid positions in get_hyperedge_styling_data."""
        # Position dict missing some nodes
        pos = {1: (0, 0), 2: (1, 0)}  # Missing node 3
        hyperedge = (1, 2, 3)
        
        # Should raise KeyError when trying to access missing node position
        with pytest.raises(KeyError):
            visualizer.get_hyperedge_styling_data(
                hyperedge, pos
            )

    def test_get_hyperedge_labels_filters_by_size(self, hypergraph_with_metadata):
        """Test that get_hyperedge_labels only returns edges with size > 2."""
        visualizer = HypergraphVisualizer(hypergraph_with_metadata)
        labels = visualizer.get_hyperedge_labels("type")
        
        # All returned edges should have size > 2 (be hyperedges)
        for edge in labels.keys():
            assert len(edge) > 2

    def test_get_hyperedge_labels_with_no_matching_key(self, hypergraph_with_metadata):
        """Test get_hyperedge_labels when no edges have the requested metadata key."""
        visualizer = HypergraphVisualizer(hypergraph_with_metadata)
        labels = visualizer.get_hyperedge_labels("nonexistent_key")
        
        # Should return empty dict since no edges have this key
        assert isinstance(labels, dict)
        assert len(labels) == 0

    @pytest.mark.parametrize("hyperedge_size", [3, 4, 5, 6])
    def test_get_hyperedge_styling_data_various_sizes(self, hyperedge_size):
        """Test get_hyperedge_styling_data with various hyperedge sizes."""
        # Create a hypergraph with an edge of the specified size
        h = Hypergraph()
        nodes = list(range(hyperedge_size))
        edge = tuple(nodes)
        h.add_nodes(nodes)
        h.add_edge(edge)
        
        visualizer = HypergraphVisualizer(h)
        
        # Create positions for all nodes
        pos = {i: (i, 0) for i in nodes}
        
        # Should not raise errors for any reasonable size
        result = visualizer.get_hyperedge_styling_data(
            edge, pos
        )
        
        assert len(result) == 2
        x_coords, y_coords = result
        assert isinstance(x_coords, list)
        assert isinstance(y_coords, list)
        assert len(x_coords) > 0
        assert len(y_coords) > 0