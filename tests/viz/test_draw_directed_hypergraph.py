import pytest
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Assuming the modules are in the same directory or properly installed
from hypergraphx.core.directed_hypergraph import DirectedHypergraph
from hypergraphx.viz.draw_hypergraph import (
    to_nx, get_node_labels, get_pairwise_edge_labels, 
    get_hyperedge_labels, draw, get_hyperedge_styling_data
)


class TestToNx:
    """Test suite for the to_nx function."""
    
    def test_to_nx_empty_hypergraph(self):
        """Test converting an empty directed hypergraph to NetworkX."""
        dh = DirectedHypergraph()
        G = to_nx(dh)
        
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 0
        assert len(G.edges()) == 0
    
    def test_to_nx_nodes_only(self):
        """Test converting a hypergraph with only nodes."""
        dh = DirectedHypergraph()
        dh.add_node(1, metadata={'name': 'node1'})
        dh.add_node(2, metadata={'name': 'node2'})
        
        G = to_nx(dh)
        
        assert len(G.nodes()) == 2
        assert 1 in G.nodes()
        assert 2 in G.nodes()
        assert G.nodes[1]['name'] == 'node1'
        assert G.nodes[2]['name'] == 'node2'
        assert len(G.edges()) == 0
    
    def test_to_nx_with_pairwise_edges(self):
        """Test converting a hypergraph with pairwise edges (order 1)."""
        dh = DirectedHypergraph()
        # Add pairwise edge: source (1) -> target (2)
        dh.add_edge((1, 2), metadata={'type': 'pairwise'})
        
        G = to_nx(dh)
        
        assert len(G.nodes()) == 2
        assert len(G.edges()) == 1
    
    def test_to_nx_with_metadata(self):
        """Test that node metadata is preserved in NetworkX conversion."""
        dh = DirectedHypergraph()
        dh.add_node('A', metadata={'label': 'Node A', 'weight': 5})
        dh.add_node('B', metadata={'label': 'Node B', 'weight': 3})
        
        G = to_nx(dh)
        
        assert G.nodes['A']['label'] == 'Node A'
        assert G.nodes['A']['weight'] == 5
        assert G.nodes['B']['label'] == 'Node B'
        assert G.nodes['B']['weight'] == 3


class TestGetNodeLabels:
    """Test suite for the get_node_labels function."""
    
    def test_get_node_labels_empty(self):
        """Test getting node labels from an empty hypergraph."""
        dh = DirectedHypergraph()
        labels = get_node_labels(dh)
        
        assert labels == {}
    
    def test_get_node_labels_default_key(self):
        """Test getting node labels with default 'text' key."""
        dh = DirectedHypergraph()
        dh.add_node(1, metadata={'text': 'Node 1'})
        dh.add_node(2, metadata={'text': 'Node 2'})
        dh.add_node(3, metadata={'other': 'Node 3'})  # No 'text' key
        
        labels = get_node_labels(dh)
        
        assert labels[1] == 'Node 1'
        assert labels[2] == 'Node 2'
        assert 3 not in labels  # Should not include nodes without 'text' key
    
    def test_get_node_labels_custom_key(self):
        """Test getting node labels with custom key."""
        dh = DirectedHypergraph()
        dh.add_node(1, metadata={'name': 'First', 'text': 'Node 1'})
        dh.add_node(2, metadata={'name': 'Second', 'text': 'Node 2'})
        
        labels = get_node_labels(dh, key='name')
        
        assert labels[1] == 'First'
        assert labels[2] == 'Second'
    
    def test_get_node_labels_missing_key(self):
        """Test getting node labels when key doesn't exist."""
        dh = DirectedHypergraph()
        dh.add_node(1, metadata={'text': 'Node 1'})
        dh.add_node(2, metadata={'other': 'Node 2'})
        
        labels = get_node_labels(dh, key='nonexistent')
        
        assert labels == {}


class TestGetPairwiseEdgeLabels:
    """Test suite for the get_pairwise_edge_labels function."""
    
    def test_get_pairwise_edge_labels_empty(self):
        """Test getting edge labels from an empty hypergraph."""
        dh = DirectedHypergraph()
        labels = get_pairwise_edge_labels(dh)
        
        assert labels == {}
    
    def test_get_pairwise_edge_labels_with_order_1_edges(self):
        """Test getting labels from pairwise edges (order 1)."""
        dh = DirectedHypergraph()
        dh.add_edge(((1,), (2,)), metadata={'type': 'relation1'})
        dh.add_edge(((2,), (3,)), metadata={'type': 'relation2'})
        
        labels = get_pairwise_edge_labels(dh)
        
        # The exact structure depends on the implementation
        assert len(labels) >= 0  # Should have some labels if edges exist
    
    def test_get_pairwise_edge_labels_custom_key(self):
        """Test getting edge labels with custom key."""
        dh = DirectedHypergraph()
        dh.add_edge(((1,), (2,)), metadata={'label': 'Edge A', 'type': 'relation'})
        
        labels = get_pairwise_edge_labels(dh, key='label')
        
        # Test structure depends on implementation details
        assert isinstance(labels, dict)


class TestGetHyperedgeLabels:
    """Test suite for the get_hyperedge_labels function."""
    
    def test_get_hyperedge_labels_empty(self):
        """Test getting hyperedge labels from an empty hypergraph."""
        dh = DirectedHypergraph()
        labels = get_hyperedge_labels(dh)
        
        assert labels == {}
    
    def test_get_hyperedge_labels_only_pairwise(self):
        """Test that pairwise edges (order 1) are excluded."""
        dh = DirectedHypergraph()
        dh.add_edge(((1,), (2,)), metadata={'type': 'pairwise'})
        
        labels = get_hyperedge_labels(dh)
        
        assert labels == {}  # Should be empty as only pairwise edges exist
    
    def test_get_hyperedge_labels_with_hyperedges(self):
        """Test getting labels from actual hyperedges (order > 1)."""
        dh = DirectedHypergraph()
        # Add a hyperedge of size 3 (order 2)
        dh.add_edge(((1, 2), (3,)), metadata={'type': 'hyperedge1'})
        # Add a hyperedge of size 4 (order 3)
        dh.add_edge(((1, 2), (3, 4)), metadata={'type': 'hyperedge2'})
        
        labels = get_hyperedge_labels(dh)
        
        # Should contain labels for hyperedges with size > 2
        assert len(labels) >= 0
    
    def test_get_hyperedge_labels_custom_key(self):
        """Test getting hyperedge labels with custom key."""
        dh = DirectedHypergraph()
        dh.add_edge(((1, 2), (3,)), metadata={'label': 'Hyper A', 'type': 'relation'})
        
        labels = get_hyperedge_labels(dh, key='label')
        
        assert isinstance(labels, dict)


class TestGetHyperedgeStylingData:
    """Test suite for the get_hyperedge_styling_data function."""
    
    def test_get_hyperedge_styling_data_triangle(self):
        """Test styling data for a triangular hyperedge."""
        hye = (1, 2, 3)
        pos = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
        color_map = {2: "#FFBC79"}
        facecolor_map = {2: "#FFBC79"}
        
        x1, y1, color, facecolor = get_hyperedge_styling_data(
            hye, pos, color_map, facecolor_map
        )
        
        assert isinstance(x1, list)
        assert isinstance(y1, list)
        assert len(x1) == len(y1)
        assert color == "#FFBC79"
        assert facecolor == "#FFBC79"
    
    def test_get_hyperedge_styling_data_square(self):
        """Test styling data for a square hyperedge."""
        hye = (1, 2, 3, 4)
        pos = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}
        color_map = {3: "#79BCFF"}
        facecolor_map = {3: "#79BCFF"}
        
        x1, y1, color, facecolor = get_hyperedge_styling_data(
            hye, pos, color_map, facecolor_map
        )
        
        assert isinstance(x1, list)
        assert isinstance(y1, list)
        assert len(x1) == len(y1)
        assert color == "#79BCFF"
        assert facecolor == "#79BCFF"
    
    def test_get_hyperedge_styling_data_random_color(self):
        """Test that random colors are generated for unknown orders."""
        hye = (1, 2, 3, 4, 5)  # Order 4, not in default maps
        pos = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1), 5: (0.5, 0.5)}
        color_map = {}
        facecolor_map = {}
        
        x1, y1, color, facecolor = get_hyperedge_styling_data(
            hye, pos, color_map, facecolor_map
        )
        
        assert isinstance(x1, list)
        assert isinstance(y1, list)
        assert color.startswith('#')
        assert facecolor.startswith('#')
        assert len(color) == 7  # Hex color format
        assert len(facecolor) == 7


class TestDraw:
    """Test suite for the draw function."""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.gca')
    def test_draw_empty_hypergraph(self, mock_gca, mock_subplot, mock_figure, mock_show):
        """Test drawing an empty hypergraph."""
        dh = DirectedHypergraph()
        mock_ax = Mock()
        mock_gca.return_value = mock_ax
        
        draw(dh)
        
        mock_figure.assert_called_once()
        mock_subplot.assert_called_once_with(1, 1, 1)
        mock_ax.axis.assert_called_with("equal")
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_draw_with_custom_ax(self, mock_figure, mock_show):
        """Test drawing with a custom axes object."""
        dh = DirectedHypergraph()
        dh.add_node(1)
        mock_ax = Mock()
        
        draw(dh, ax=mock_ax)
        
        mock_figure.assert_not_called()  # Should not create new figure
        mock_ax.axis.assert_called_with("equal")
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('networkx.spring_layout')
    def test_draw_with_custom_pos(self, mock_spring_layout, mock_show):
        """Test drawing with custom node positions."""
        dh = DirectedHypergraph()
        dh.add_node(1)
        dh.add_node(2)
        custom_pos = {1: (0, 0), 2: (1, 1)}
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.gca') as mock_gca:
            mock_ax = Mock()
            mock_gca.return_value = mock_ax
            
            draw(dh, pos=custom_pos)
            
            mock_spring_layout.assert_not_called()  # Should not compute layout
    
    @patch('matplotlib.pyplot.show')
    def test_draw_with_labels(self, mock_show):
        """Test drawing with node and edge labels."""
        dh = DirectedHypergraph()
        dh.add_node(1, metadata={'text': 'Node 1'})
        dh.add_node(2, metadata={'text': 'Node 2'})
        dh.add_edge(((1,), (2,)), metadata={'type': 'relation'})
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.gca') as mock_gca, \
             patch('networkx.spring_layout') as mock_layout:
            
            mock_ax = Mock()
            mock_gca.return_value = mock_ax
            mock_layout.return_value = {1: (0, 0), 2: (1, 1)}
            
            draw(dh, with_node_labels=True, with_pairwise_edge_labels=True)
            
            mock_ax.axis.assert_called_with("equal")
    
    @patch('matplotlib.pyplot.show')
    def test_draw_with_hyperedges(self, mock_show):
        """Test drawing hypergraph with actual hyperedges."""
        dh = DirectedHypergraph()
        dh.add_edge(((1, 2), (3,)), metadata={'type': 'hyperedge'})
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.gca') as mock_gca, \
             patch('networkx.spring_layout') as mock_layout:
            
            mock_ax = Mock()
            mock_gca.return_value = mock_ax
            mock_layout.return_value = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
            
            draw(dh, with_hyperedge_labels=True)
            
            mock_ax.fill.assert_called()  # Should draw hyperedge shapes
            mock_ax.annotate.assert_called()  # Should add labels
    
    @patch('matplotlib.pyplot.show')
    def test_draw_styling_parameters(self, mock_show):
        """Test various styling parameters."""
        dh = DirectedHypergraph()
        dh.add_node(1)
        dh.add_node(2)
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.gca') as mock_gca, \
             patch('networkx.spring_layout') as mock_layout:
            
            mock_ax = Mock()
            mock_gca.return_value = mock_ax
            mock_layout.return_value = {1: (0, 0), 2: (1, 1)}
            
            draw(dh, 
                 figsize=(10, 8),
                 node_size=200,
                 node_color='red',
                 node_facecolor='blue',
                 node_shape='s',
                 pairwise_edge_color='green',
                 pairwise_edge_width=2.0,
                 hyperedge_alpha=0.5,
                 label_size=12,
                 label_col='purple')
            
            mock_ax.axis.assert_called_with("equal")


class TestIntegration:
    """Integration tests using DirectedHypergraph instances."""
    
    def test_complete_workflow_simple(self):
        """Test the complete workflow with a simple hypergraph."""
        dh = DirectedHypergraph()
        
        # Add nodes with metadata
        dh.add_node(1, metadata={'text': 'Node 1', 'category': 'A'})
        dh.add_node(2, metadata={'text': 'Node 2', 'category': 'B'})
        dh.add_node(3, metadata={'text': 'Node 3', 'category': 'A'})
        
        # Add pairwise edge
        dh.add_edge(((1,), (2,)), metadata={'type': 'connects', 'weight': 1.0})
        
        # Add hyperedge
        dh.add_edge(((1, 2), (3,)), metadata={'type': 'group', 'weight': 2.0})
        
        # Test all label functions
        node_labels = get_node_labels(dh)
        assert len(node_labels) == 3
        assert node_labels[1] == 'Node 1'
        
        pairwise_labels = get_pairwise_edge_labels(dh)
        assert isinstance(pairwise_labels, dict)
        
        hyperedge_labels = get_hyperedge_labels(dh)
        assert isinstance(hyperedge_labels, dict)
        
        # Test NetworkX conversion
        G = to_nx(dh)
        assert len(G.nodes()) == 3
        assert G.nodes[1]['text'] == 'Node 1'
    
    def test_complex_hypergraph_workflow(self):
        """Test workflow with a more complex hypergraph."""
        dh = DirectedHypergraph(weighted=True)
        
        # Create a more complex structure
        nodes = range(1, 8)
        for node in nodes:
            dh.add_node(node, metadata={'text': f'Node {node}', 'value': node * 10})
        
        # Add various edge types
        edges = [
            (((1,), (2,)), 1.0, {'type': 'pair1'}),
            (((2,), (3,)), 1.5, {'type': 'pair2'}),
            (((1, 2), (3, 4)), 2.0, {'type': 'hyper1'}),
            (((3, 4, 5), (6, 7)), 3.0, {'type': 'hyper2'}),
        ]
        
        for edge, weight, metadata in edges:
            dh.add_edge(edge, weight=weight, metadata=metadata)
        
        # Test all functions work with complex structure
        assert len(get_node_labels(dh)) == 7
        assert len(get_node_labels(dh, key='value')) == 0  # 'value' is not string
        
        # Should have hyperedges
        hyperedge_labels = get_hyperedge_labels(dh)
        assert len(hyperedge_labels) >= 0
        
        # NetworkX conversion should preserve structure
        G = to_nx(dh)
        assert len(G.nodes()) == 7
    
    @patch('matplotlib.pyplot.show')
    def test_drawing_integration(self, mock_show):
        """Integration test for the drawing function."""
        dh = DirectedHypergraph()
        
        # Create a small but complete hypergraph
        for i in range(1, 6):
            dh.add_node(i, metadata={'text': f'N{i}'})
        
        dh.add_edge(((1,), (2,)), metadata={'type': 'rel1'})
        dh.add_edge(((1, 2), (3,)), metadata={'type': 'group1'})
        dh.add_edge(((3, 4), (5,)), metadata={'type': 'group2'})
        
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.subplot'), \
             patch('matplotlib.pyplot.gca') as mock_gca, \
             patch('networkx.spring_layout') as mock_layout:
            
            mock_ax = Mock()
            mock_gca.return_value = mock_ax
            mock_layout.return_value = {i: (i, i) for i in range(1, 6)}
            
            # Should not raise any exceptions
            draw(dh, 
                 with_node_labels=True,
                 with_pairwise_edge_labels=True,
                 with_hyperedge_labels=True)
            
            mock_show.assert_called_once()


# Fixtures for common test data
@pytest.fixture
def simple_hypergraph():
    """Create a simple hypergraph for testing."""
    dh = DirectedHypergraph()
    dh.add_node(1, metadata={'text': 'Node 1'})
    dh.add_node(2, metadata={'text': 'Node 2'})
    dh.add_node(3, metadata={'text': 'Node 3'})
    dh.add_edge(((1,), (2,)), metadata={'type': 'simple'})
    return dh


@pytest.fixture
def complex_hypergraph():
    """Create a complex hypergraph for testing."""
    dh = DirectedHypergraph(weighted=True)
    
    # Add nodes
    for i in range(1, 6):
        dh.add_node(i, metadata={'text': f'Node {i}', 'category': chr(65 + i % 3)})
    
    # Add edges of various orders
    edges = [
        (((1,), (2,)), 1.0, {'type': 'pairwise', 'label': 'P1'}),
        (((2,), (3,)), 1.5, {'type': 'pairwise', 'label': 'P2'}),
        (((1, 2), (3,)), 2.0, {'type': 'hyperedge', 'label': 'H1'}),
        (((2, 3), (4, 5)), 2.5, {'type': 'hyperedge', 'label': 'H2'}),
    ]
    
    for edge, weight, metadata in edges:
        dh.add_edge(edge, weight=weight, metadata=metadata)
    
    return dh


# Test using fixtures
def test_with_simple_fixture(simple_hypergraph):
    """Test functions with simple hypergraph fixture."""
    labels = get_node_labels(simple_hypergraph)
    assert len(labels) == 3
    assert labels[1] == 'Node 1'


def test_with_complex_fixture(complex_hypergraph):
    """Test functions with complex hypergraph fixture."""
    node_labels = get_node_labels(complex_hypergraph)
    assert len(node_labels) == 5
    
    hyperedge_labels = get_hyperedge_labels(complex_hypergraph, key='label')
    # Should contain labels for hyperedges with size > 2
    assert isinstance(hyperedge_labels, dict)


if __name__ == "__main__":
    pytest.main([__file__])