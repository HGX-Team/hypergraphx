def hypergraph_to_dot(edgelist, method="auxiliary", graph_name="hypergraph"):
    """
    Convert a hypergraph edgelist to DOT format.
    
    Args:
        edgelist: List of tuples or single tuples
                 - Directed: ((sources,), (targets,))
                 - Undirected: (node1, node2, node3, ...)
        method: "auxiliary", "direct", or "cluster" 
        graph_name: Name for the DOT graph
    
    Returns:
        String containing DOT format representation
    """
    
    # Separate directed and undirected hyperedges
    directed_edges, undirected_edges = _parse_edgelist(edgelist)
    
    if method == "auxiliary":
        return _auxiliary_method(directed_edges, undirected_edges, graph_name)
    elif method == "direct":
        return _direct_method(directed_edges, undirected_edges, graph_name)
    elif method == "cluster":
        return _cluster_method(directed_edges, undirected_edges, graph_name)
    else:
        raise ValueError("Method must be 'auxiliary', 'direct', or 'cluster'")

def _parse_edgelist(edgelist):
    """
    Parse edgelist to separate directed and undirected hyperedges.
    
    Returns:
        tuple: (directed_edges, undirected_edges)
    """
    directed_edges = []
    undirected_edges = []
    
    for edge in edgelist:
        if isinstance(edge, tuple) and len(edge) == 2 and isinstance(edge[0], tuple):
            # Directed hyperedge: ((sources,), (targets,))
            directed_edges.append(edge)
        else:
            # Undirected hyperedge: (node1, node2, node3, ...)
            if isinstance(edge, tuple) and len(edge) > 1:
                undirected_edges.append(edge)
            else:
                raise ValueError(f"Invalid edge format: {edge}")
    
    return directed_edges, undirected_edges

def _auxiliary_method(directed_edges, undirected_edges, graph_name):
    """Convert using auxiliary hyperedge nodes with clusters for multi-node groups."""
    
    # Collect all nodes
    all_nodes = set()
    for sources, targets in directed_edges:
        all_nodes.update(sources)
        all_nodes.update(targets)
    for nodes in undirected_edges:
        all_nodes.update(nodes)
    
    dot_lines = [f"digraph {graph_name} {{"]
    dot_lines.append("    // Global node styling")
    dot_lines.append("    node [shape=circle];")
    dot_lines.append("")
    
    # Track which nodes are in clusters to avoid duplication
    clustered_nodes = set()
    cluster_counter = 0
    
    # Create clusters for multi-node source/target groups in directed edges
    if directed_edges:
        dot_lines.append("    // Source and target clusters for directed hyperedges")
        
        for i, (sources, targets) in enumerate(directed_edges):
            # Create source cluster if multiple sources
            if len(sources) > 1:
                cluster_name = f"cluster_src_{i+1}"
                dot_lines.append(f"    subgraph {cluster_name} {{")
                dot_lines.append(f'        label="Sources {i+1}";')
                dot_lines.append("        style=filled;")
                dot_lines.append("        fillcolor=lightcyan;")
                dot_lines.append("        color=blue;")
                for source in sorted(sources):
                    dot_lines.append(f"        {source};")
                    clustered_nodes.add(source)
                dot_lines.append("    }")
                dot_lines.append("")
            
            # Create target cluster if multiple targets
            if len(targets) > 1:
                cluster_name = f"cluster_tgt_{i+1}"
                dot_lines.append(f"    subgraph {cluster_name} {{")
                dot_lines.append(f'        label="Targets {i+1}";')
                dot_lines.append("        style=filled;")
                dot_lines.append("        fillcolor=lightpink;")
                dot_lines.append("        color=red;")
                for target in sorted(targets):
                    dot_lines.append(f"        {target};")
                    clustered_nodes.add(target)
                dot_lines.append("    }")
                dot_lines.append("")
    
    # Create clusters for undirected hyperedges
    if undirected_edges:
        dot_lines.append("    // Undirected hyperedge clusters")
        for i, nodes in enumerate(undirected_edges):
            cluster_name = f"cluster_undirected_{i+1}"
            dot_lines.append(f"    subgraph {cluster_name} {{")
            dot_lines.append(f'        label="Undirected HE {i+1}";')
            dot_lines.append("        style=filled;")
            dot_lines.append("        fillcolor=lightyellow;")
            dot_lines.append("        color=orange;")
            for node in sorted(nodes):
                dot_lines.append(f"        {node};")
                clustered_nodes.add(node)
            # Add internal connections within undirected hyperedge
            nodes_list = list(nodes)
            if len(nodes_list) > 1:
                dot_lines.append("")
                for j in range(len(nodes_list)):
                    for k in range(j + 1, len(nodes_list)):
                        dot_lines.append(f"        {nodes_list[j]} -> {nodes_list[k]} [dir=none, color=orange];")
            dot_lines.append("    }")
            dot_lines.append("")
    
    # Add standalone nodes (not in any cluster)
    standalone_nodes = all_nodes - clustered_nodes
    if standalone_nodes:
        dot_lines.append("    // Standalone nodes")
        for node in sorted(standalone_nodes):
            dot_lines.append(f"    {node};")
        dot_lines.append("")
    
    # Add directed hyperedge auxiliary nodes
    if directed_edges:
        dot_lines.append("    // Directed hyperedge auxiliary nodes")
        for i, (sources, targets) in enumerate(directed_edges):
            he_name = f"dhe{i+1}"
            dot_lines.append(f'    {he_name} [label="DHE{i+1}", shape=box, style=filled, fillcolor=lightblue];')
        dot_lines.append("")
        
        dot_lines.append("    // Directed hyperedge connections")
        for i, (sources, targets) in enumerate(directed_edges):
            he_name = f"dhe{i+1}"
            
            # Sources to hyperedge
            for source in sources:
                dot_lines.append(f"    {source} -> {he_name};")
            
            # Hyperedge to targets
            for target in targets:
                dot_lines.append(f"    {he_name} -> {target};")
            
            dot_lines.append("")
    
    # Add undirected hyperedge auxiliary nodes
    if undirected_edges:
        dot_lines.append("    // Undirected hyperedge auxiliary nodes")
        for i, nodes in enumerate(undirected_edges):
            he_name = f"uhe{i+1}"
            dot_lines.append(f'    {he_name} [label="UHE{i+1}", shape=diamond, style=filled, fillcolor=lightgreen];')
        dot_lines.append("")
        
        dot_lines.append("    // Undirected hyperedge connections")
        for i, nodes in enumerate(undirected_edges):
            he_name = f"uhe{i+1}"
            
            # Bidirectional connections between hyperedge and all nodes
            for node in nodes:
                dot_lines.append(f"    {node} -> {he_name} [dir=none];")
            
            dot_lines.append("")
    
    dot_lines.append("}")
    return "\n".join(dot_lines)

def _direct_method(directed_edges, undirected_edges, graph_name):
    """Convert using direct connections with clusters for multi-node groups."""
    
    # Collect all nodes
    all_nodes = set()
    for sources, targets in directed_edges:
        all_nodes.update(sources)
        all_nodes.update(targets)
    for nodes in undirected_edges:
        all_nodes.update(nodes)
    
    # Track which nodes are in clusters to avoid duplication
    clustered_nodes = set()
    direct_edges_set = set()
    
    dot_lines = [f"digraph {graph_name} {{"]
    dot_lines.append("    // Global node styling")
    dot_lines.append("    node [shape=circle];")
    dot_lines.append("")
    
    # Create clusters for multi-node source/target groups in directed edges
    if directed_edges:
        dot_lines.append("    // Source and target clusters for directed hyperedges")
        
        for i, (sources, targets) in enumerate(directed_edges):
            # Create source cluster if multiple sources
            if len(sources) > 1:
                cluster_name = f"cluster_src_{i+1}"
                dot_lines.append(f"    subgraph {cluster_name} {{")
                dot_lines.append(f'        label="Sources {i+1}";')
                dot_lines.append("        style=filled;")
                dot_lines.append("        fillcolor=lightcyan;")
                dot_lines.append("        color=blue;")
                for source in sorted(sources):
                    dot_lines.append(f"        {source};")
                    clustered_nodes.add(source)
                
                # Add internal connections within source cluster
                sources_list = list(sources)
                if len(sources_list) > 1:
                    dot_lines.append("")
                    for j in range(len(sources_list)):
                        for k in range(j + 1, len(sources_list)):
                            dot_lines.append(f"        {sources_list[j]} -> {sources_list[k]} [dir=none, color=blue, style=dashed];")
                
                dot_lines.append("    }")
                dot_lines.append("")
            
            # Create target cluster if multiple targets
            if len(targets) > 1:
                cluster_name = f"cluster_tgt_{i+1}"
                dot_lines.append(f"    subgraph {cluster_name} {{")
                dot_lines.append(f'        label="Targets {i+1}";')
                dot_lines.append("        style=filled;")
                dot_lines.append("        fillcolor=lightpink;")
                dot_lines.append("        color=red;")
                for target in sorted(targets):
                    dot_lines.append(f"        {target};")
                    clustered_nodes.add(target)
                
                # Add internal connections within target cluster
                targets_list = list(targets)
                if len(targets_list) > 1:
                    dot_lines.append("")
                    for j in range(len(targets_list)):
                        for k in range(j + 1, len(targets_list)):
                            dot_lines.append(f"        {targets_list[j]} -> {targets_list[k]} [dir=none, color=red, style=dashed];")
                
                dot_lines.append("    }")
                dot_lines.append("")
            
            # Create direct edges from each source to each target
            for source in sources:
                for target in targets:
                    direct_edges_set.add((source, target))
    
    # Create clusters for undirected hyperedges
    if undirected_edges:
        dot_lines.append("    // Undirected hyperedge clusters")
        for i, nodes in enumerate(undirected_edges):
            cluster_name = f"cluster_undirected_{i+1}"
            dot_lines.append(f"    subgraph {cluster_name} {{")
            dot_lines.append(f'        label="Undirected HE {i+1}";')
            dot_lines.append("        style=filled;")
            dot_lines.append("        fillcolor=lightyellow;")
            dot_lines.append("        color=orange;")
            
            for node in sorted(nodes):
                dot_lines.append(f"        {node};")
                clustered_nodes.add(node)
            
            # Add bidirectional edges between all pairs in the hyperedge
            nodes_list = list(nodes)
            if len(nodes_list) > 1:
                dot_lines.append("")
                for j in range(len(nodes_list)):
                    for k in range(j + 1, len(nodes_list)):
                        dot_lines.append(f"        {nodes_list[j]} -> {nodes_list[k]} [dir=none, color=orange];")
            
            dot_lines.append("    }")
            dot_lines.append("")
    
    # Add standalone nodes (not in any cluster)
    standalone_nodes = all_nodes - clustered_nodes
    if standalone_nodes:
        dot_lines.append("    // Standalone nodes")
        for node in sorted(standalone_nodes):
            dot_lines.append(f"        {node};")
        dot_lines.append("")
    
    # Add directed edges between clusters and standalone nodes
    if direct_edges_set:
        dot_lines.append("    // Directed edges")
        for source, target in sorted(direct_edges_set):
            dot_lines.append(f"    {source} -> {target};")
        dot_lines.append("")
    
    dot_lines.append("}")
    return "\n".join(dot_lines)

def _cluster_method(directed_edges, undirected_edges, graph_name):
    """Convert using subgraph clusters for undirected hyperedges."""
    
    # Collect all nodes
    all_nodes = set()
    for sources, targets in directed_edges:
        all_nodes.update(sources)
        all_nodes.update(targets)
    for nodes in undirected_edges:
        all_nodes.update(nodes)
    
    # Find nodes that are only in clusters (not in directed edges)
    nodes_in_directed = set()
    for sources, targets in directed_edges:
        nodes_in_directed.update(sources)
        nodes_in_directed.update(targets)
    
    dot_lines = [f"digraph {graph_name} {{"]
    dot_lines.append("    // Global node styling")
    dot_lines.append("    node [shape=circle];")
    dot_lines.append("")
    
    # Create clusters for undirected hyperedges
    if undirected_edges:
        for i, nodes in enumerate(undirected_edges):
            cluster_name = f"cluster_{i}"
            dot_lines.append(f"    subgraph {cluster_name} {{")
            dot_lines.append(f'        label="Hyperedge {i+1}";')
            dot_lines.append("        style=filled;")
            dot_lines.append("        fillcolor=lightyellow;")
            dot_lines.append("        color=orange;")
            dot_lines.append("")
            
            # Add nodes to cluster
            for node in sorted(nodes):
                dot_lines.append(f"        {node};")
            
            # Add edges within cluster (complete subgraph)
            nodes_list = list(nodes)
            if len(nodes_list) > 1:
                dot_lines.append("")
                dot_lines.append("        // Internal connections")
                for j in range(len(nodes_list)):
                    for k in range(j + 1, len(nodes_list)):
                        dot_lines.append(f"        {nodes_list[j]} -> {nodes_list[k]} [dir=none, color=orange];")
            
            dot_lines.append("    }")
            dot_lines.append("")
    
    # Add standalone nodes (not in any cluster)
    standalone_nodes = all_nodes - set().union(*undirected_edges) if undirected_edges else all_nodes
    if standalone_nodes:
        dot_lines.append("    // Standalone nodes")
        for node in sorted(standalone_nodes):
            dot_lines.append(f"    {node};")
        dot_lines.append("")
    
    # Add directed edges
    if directed_edges:
        dot_lines.append("    // Directed edges")
        for sources, targets in directed_edges:
            for source in sources:
                for target in targets:
                    dot_lines.append(f"    {source} -> {target};")
        dot_lines.append("")
    
    dot_lines.append("}")
    return "\n".join(dot_lines)

def save_and_render(dot_content:str, filename:str, format:str = "png"):
    """
    Save DOT content to file and optionally render with Graphviz.
    
    Args:
        dot_content: DOT format string
        filename: Output filename (without extension)
        format: Output format (png, svg, pdf, etc.)
    """
    
    # Save DOT file
    dot_filename = f"{filename}.dot"
    with open(dot_filename, 'w') as f:
        f.write(dot_content)
    
    print(f"DOT file saved as: {dot_filename}")
    print(f"To render with Graphviz, run:")
    print(f"dot -T{format} {dot_filename} -o {filename}.{format}")