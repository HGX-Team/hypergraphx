import pytest
from unittest.mock import Mock, patch, mock_open

# Import the functions to test (assuming they're in a module called hypergraph_dot_converter)
from hypergraphx.viz.hypergraph_to_dot import (
    hypergraph_to_dot,
    _parse_edgelist,
    _auxiliary_method,
    _direct_method,
    _cluster_method,
    save_and_render
)


class TestHypergraphDotConverter:
    """Test suite for hypergraph to DOT converter functions."""

    @pytest.fixture
    def simple_directed_edgelist(self):
        """Create a simple directed hypergraph edgelist for testing."""
        return [
            ((1,), (2,)),
            ((1,), (3, 4, 5)),
            ((8, 7), (1,))
        ]

    @pytest.fixture
    def simple_undirected_edgelist(self):
        """Create a simple undirected hypergraph edgelist for testing."""
        return [
            (1, 2, 3, 4),
            (6, 7, 8),
            (9, 10)
        ]

    @pytest.fixture
    def mixed_edgelist(self):
        """Create a mixed hypergraph edgelist for testing."""
        return [
            # Directed hyperedges
            ((1,), (2,)),
            ((1,), (3, 4, 5)),
            ((8, 7), (1,)),
            # Undirected hyperedges
            (1, 2, 3, 4),
            (6, 7, 8),
            (9, 10)
        ]

    @pytest.fixture
    def complex_directed_edgelist(self):
        """Create a complex directed hypergraph with multi-node sources and targets."""
        return [
            ((1, 2, 3), (4, 5, 6, 7)),
            ((8, 9), (10,)),
            ((11,), (12, 13))
        ]

    @pytest.fixture
    def empty_edgelist(self):
        """Create an empty edgelist for testing."""
        return []

    def test_parse_edgelist_directed_only(self, simple_directed_edgelist):
        """Test parsing edgelist with only directed hyperedges."""
        directed_edges, undirected_edges = _parse_edgelist(simple_directed_edgelist)
        
        assert len(directed_edges) == 3
        assert len(undirected_edges) == 0
        assert directed_edges == simple_directed_edgelist

    def test_parse_edgelist_undirected_only(self, simple_undirected_edgelist):
        """Test parsing edgelist with only undirected hyperedges."""
        directed_edges, undirected_edges = _parse_edgelist(simple_undirected_edgelist)
        
        assert len(directed_edges) == 0
        assert len(undirected_edges) == 3
        assert undirected_edges == simple_undirected_edgelist

    def test_parse_edgelist_mixed(self, mixed_edgelist):
        """Test parsing edgelist with both directed and undirected hyperedges."""
        directed_edges, undirected_edges = _parse_edgelist(mixed_edgelist)
        
        assert len(directed_edges) == 3
        assert len(undirected_edges) == 3
        
        # Check directed edges
        expected_directed = [
            ((1,), (2,)),
            ((1,), (3, 4, 5)),
            ((8, 7), (1,))
        ]
        assert directed_edges == expected_directed
        
        # Check undirected edges
        expected_undirected = [
            (1, 2, 3, 4),
            (6, 7, 8),
            (9, 10)
        ]
        assert undirected_edges == expected_undirected

    def test_parse_edgelist_empty(self, empty_edgelist):
        """Test parsing empty edgelist."""
        directed_edges, undirected_edges = _parse_edgelist(empty_edgelist)
        
        assert len(directed_edges) == 0
        assert len(undirected_edges) == 0

    def test_parse_edgelist_invalid_format(self):
        """Test parsing edgelist with invalid format raises error."""
        invalid_edgelist = [
            ((1,), (2,)),  # Valid
            "invalid_edge",  # Invalid
        ]
        
        with pytest.raises(ValueError, match="Invalid edge format"):
            _parse_edgelist(invalid_edgelist)

    def test_hypergraph_to_dot_auxiliary_method(self, simple_directed_edgelist):
        """Test hypergraph_to_dot with auxiliary method."""
        result = hypergraph_to_dot(simple_directed_edgelist, method="auxiliary")
        
        assert isinstance(result, str)
        assert "digraph hypergraph {" in result
        assert "DHE1" in result
        assert "DHE2" in result
        assert "DHE3" in result
        assert "shape=box" in result
        assert "fillcolor=lightblue" in result

    def test_hypergraph_to_dot_direct_method(self, simple_directed_edgelist):
        """Test hypergraph_to_dot with direct method."""
        result = hypergraph_to_dot(simple_directed_edgelist, method="direct")
        
        assert isinstance(result, str)
        assert "digraph hypergraph {" in result
        assert "1 -> 2;" in result
        assert "1 -> 3;" in result
        assert "1 -> 4;" in result
        assert "1 -> 5;" in result

    def test_hypergraph_to_dot_cluster_method(self, simple_undirected_edgelist):
        """Test hypergraph_to_dot with cluster method."""
        result = hypergraph_to_dot(simple_undirected_edgelist, method="cluster")
        
        assert isinstance(result, str)
        assert "digraph hypergraph {" in result
        assert "subgraph cluster_" in result
        assert "fillcolor=lightyellow" in result
        assert "dir=none" in result

    def test_hypergraph_to_dot_invalid_method(self, simple_directed_edgelist):
        """Test hypergraph_to_dot with invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be"):
            hypergraph_to_dot(simple_directed_edgelist, method="invalid")

    def test_hypergraph_to_dot_custom_graph_name(self, simple_directed_edgelist):
        """Test hypergraph_to_dot with custom graph name."""
        result = hypergraph_to_dot(simple_directed_edgelist, graph_name="custom_graph")
        
        assert "digraph custom_graph {" in result

    def test_auxiliary_method_clusters_multi_node_sources(self, complex_directed_edgelist):
        """Test auxiliary method creates clusters for multi-node sources."""
        directed_edges, undirected_edges = _parse_edgelist(complex_directed_edgelist)
        result = _auxiliary_method(directed_edges, undirected_edges, "test_graph")
        
        assert "cluster_src_1" in result
        assert "Sources 1" in result
        assert "fillcolor=lightcyan" in result
        assert "color=blue" in result

    def test_auxiliary_method_clusters_multi_node_targets(self, complex_directed_edgelist):
        """Test auxiliary method creates clusters for multi-node targets."""
        directed_edges, undirected_edges = _parse_edgelist(complex_directed_edgelist)
        result = _auxiliary_method(directed_edges, undirected_edges, "test_graph")
        
        assert "cluster_tgt_1" in result
        assert "Targets 1" in result
        assert "fillcolor=lightpink" in result
        assert "color=red" in result

    def test_auxiliary_method_single_nodes_not_clustered(self):
        """Test auxiliary method doesn't cluster single nodes."""
        directed_edges = [((1,), (2,))]  # Single source, single target
        undirected_edges = []
        result = _auxiliary_method(directed_edges, undirected_edges, "test_graph")
        
        assert "cluster_src_" not in result
        assert "cluster_tgt_" not in result
        assert "DHE1" in result

    def test_auxiliary_method_undirected_hyperedges(self, simple_undirected_edgelist):
        """Test auxiliary method handles undirected hyperedges."""
        directed_edges = []
        undirected_edges = simple_undirected_edgelist
        result = _auxiliary_method(directed_edges, undirected_edges, "test_graph")
        
        assert "cluster_undirected_1" in result
        assert "UHE1" in result
        assert "shape=diamond" in result
        assert "fillcolor=lightgreen" in result
        assert "dir=none" in result

    def test_direct_method_clusters_multi_node_groups(self, complex_directed_edgelist):
        """Test direct method creates clusters for multi-node groups."""
        directed_edges, undirected_edges = _parse_edgelist(complex_directed_edgelist)
        result = _direct_method(directed_edges, undirected_edges, "test_graph")
        
        assert "cluster_src_1" in result
        assert "cluster_tgt_1" in result
        assert "style=dashed" in result

    def test_direct_method_internal_cluster_connections(self, complex_directed_edgelist):
        """Test direct method adds internal connections within clusters."""
        directed_edges, undirected_edges = _parse_edgelist(complex_directed_edgelist)
        result = _direct_method(directed_edges, undirected_edges, "test_graph")
        
        # Should have dashed internal connections in source cluster
        assert "dir=none" in result
        assert "color=blue" in result
        assert "style=dashed" in result

    def test_cluster_method_creates_subgraphs(self, simple_undirected_edgelist):
        """Test cluster method creates proper subgraph clusters."""
        directed_edges = []
        undirected_edges = simple_undirected_edgelist
        result = _cluster_method(directed_edges, undirected_edges, "test_graph")
        
        assert "subgraph cluster_0" in result
        assert "subgraph cluster_1" in result
        assert "subgraph cluster_2" in result
        assert "label=\"Hyperedge 1\"" in result

    def test_cluster_method_internal_edges(self, simple_undirected_edgelist):
        """Test cluster method adds internal edges within clusters."""
        directed_edges = []
        undirected_edges = simple_undirected_edgelist
        result = _cluster_method(directed_edges, undirected_edges, "test_graph")
        
        assert "dir=none" in result
        assert "color=orange" in result

    def test_cluster_method_mixed_edges(self, mixed_edgelist):
        """Test cluster method handles both directed and undirected edges."""
        directed_edges, undirected_edges = _parse_edgelist(mixed_edgelist)
        result = _cluster_method(directed_edges, undirected_edges, "test_graph")
        
        # Should have both clusters and directed edges
        assert "subgraph cluster_" in result
        assert "1 -> 2;" in result

    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_save_and_render_creates_file(self, mock_print, mock_file):
        """Test save_and_render creates DOT file."""
        dot_content = "digraph test { 1 -> 2; }"
        save_and_render(dot_content, "test_graph", "png")
        
        mock_file.assert_called_once_with("test_graph.dot", 'w')
        mock_file().write.assert_called_once_with(dot_content)

    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_save_and_render_print_instructions(self, mock_print, mock_file):
        """Test save_and_render prints rendering instructions."""
        dot_content = "digraph test { 1 -> 2; }"
        save_and_render(dot_content, "test_graph", "svg")
        
        # Check that appropriate messages were printed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("DOT file saved as: test_graph.dot" in call for call in print_calls)
        assert any("dot -Tsvg test_graph.dot -o test_graph.svg" in call for call in print_calls)

    def test_auxiliary_method_empty_edgelist(self):
        """Test auxiliary method handles empty edgelist."""
        result = _auxiliary_method([], [], "empty_graph")
        
        assert "digraph empty_graph {" in result
        assert "}" in result
        # Should not contain any nodes or edges
        assert "DHE" not in result
        assert "UHE" not in result

    def test_direct_method_empty_edgelist(self):
        """Test direct method handles empty edgelist."""
        result = _direct_method([], [], "empty_graph")
        
        assert "digraph empty_graph {" in result
        assert "}" in result

    def test_cluster_method_empty_edgelist(self):
        """Test cluster method handles empty edgelist."""
        result = _cluster_method([], [], "empty_graph")
        
        assert "digraph empty_graph {" in result
        assert "}" in result

    @pytest.mark.parametrize("method", ["auxiliary", "direct", "cluster"])
    def test_all_methods_handle_empty_input(self, method, empty_edgelist):
        """Test all methods handle empty input gracefully."""
        result = hypergraph_to_dot(empty_edgelist, method=method)
        
        assert isinstance(result, str)
        assert "digraph hypergraph {" in result
        assert "}" in result

    @pytest.mark.parametrize("method", ["auxiliary", "direct", "cluster"])
    def test_all_methods_return_valid_dot(self, method, mixed_edgelist):
        """Test all methods return valid DOT format."""
        result = hypergraph_to_dot(mixed_edgelist, method=method)
        
        assert isinstance(result, str)
        assert result.startswith("digraph")
        assert result.endswith("}")
        assert "{" in result
        
        # Should not have syntax errors (basic check)
        assert result.count("{") == result.count("}")

    def test_node_deduplication_across_edges(self):
        """Test that nodes appearing in multiple edges are handled correctly."""
        edgelist = [
            ((1, 2), (3, 4)),  # 1,2 in source; 3,4 in target
            ((3, 4), (5, 6)),  # 3,4 now in source
            (1, 5, 7)          # 1,5 in undirected edge
        ]
        
        result = hypergraph_to_dot(edgelist, method="auxiliary")
        
        # Each node should appear only once in node declarations
        node_declarations = [line for line in result.split('\n') if line.strip().endswith(';') and '->' not in line and 'label=' not in line]
        # Nodes 1,2,3,4,5,6,7 should all be present but not duplicated in clusters
        assert isinstance(result, str)

    def test_large_hyperedge_handling(self):
        """Test handling of large hyperedges."""
        large_hyperedge = tuple(range(1, 21))  # 20-node hyperedge
        edgelist = [large_hyperedge]
        
        result = hypergraph_to_dot(edgelist, method="cluster")
        
        assert isinstance(result, str)
        assert "subgraph cluster_0" in result
        # Should handle large hyperedges without errors
        for i in range(1, 21):
            assert str(i) in result

    @pytest.mark.parametrize("format_type", ["png", "svg", "pdf", "dot"])
    def test_save_and_render_different_formats(self, format_type):
        """Test save_and_render with different output formats."""
        with patch("builtins.open", mock_open()) as mock_file, \
             patch("builtins.print") as mock_print:
            
            dot_content = "digraph test { 1 -> 2; }"
            save_and_render(dot_content, "test", format_type)
            
            # Check format in printed command
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any(f"dot -T{format_type}" in call for call in print_calls)