import networkx as nx
import numpy as np

def create_grid_network(rows=5, cols=5, base_min=5.0, base_max=15.0, seed=42):
    """
    Creates a directed grid graph for the routing environment.
    Adds cross-edges (diagonals) to create more path variations.
    """
    np.random.seed(seed)
    
    # Create undirected grid
    G_undirected = nx.grid_2d_graph(rows, cols)
    
    # Convert to directed
    G = nx.DiGraph()
    for u, v in G_undirected.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)
        
    # Add diagonals for more routing options
    for r in range(rows - 1):
        for c in range(cols - 1):
            u = (r, c)
            v_diag1 = (r+1, c+1)
            v_diag2 = (r+1, c) # vertical
            u_diag2 = (r, c+1) # horizontal
            
            G.add_edge(u, v_diag1)
            G.add_edge(v_diag1, u)
            G.add_edge((r+1, c), (r, c+1))
            G.add_edge((r, c+1), (r+1, c))
            
    # Attach pos attribute before relabeling
    for node in G.nodes():
        G.nodes[node]['pos'] = node
        
    # Assign integer IDs to nodes for simpler indexing
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Assign base weights
    for u, v in G.edges():
        G.edges[u, v]['base_weight'] = np.random.uniform(base_min, base_max)
        G.edges[u, v]['weight'] = G.edges[u, v]['base_weight']
        
    return G
