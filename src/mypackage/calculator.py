import os
from collections import deque
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. Closeness Centrality
def closeness_centrality_from_scratch(G):
    """
    Calculate the closeness centrality for each node in a graph from scratch.
    """
    centrality = {}
    N = G.number_of_nodes()
    H = G.reverse() if G.is_directed() else G

    for node_i in G.nodes():
        shortest_paths = nx.shortest_path_length(H, source=node_i)
        total_distance = sum(shortest_paths.values())
        n_reachable = len(shortest_paths)

        if total_distance > 0 and N > 1:
            centrality[node_i] = (n_reachable - 1) / total_distance
            centrality[node_i] *= (n_reachable - 1) / (N - 1)
        else:
            centrality[node_i] = 0.0

    return centrality


# 2. Eigenvector Centrality
def eigenvector_centrality_from_scratch(G, max_iter=100, tol=1e-08):
    """
    Calculate the eigenvector centrality for each node in a graph from scratch.
    """
    centrality = {node: 1.0 / len(G) for node in G}
    in_neighbors = G.predecessors if G.is_directed() else G.neighbors

    for _ in range(max_iter):
        prev_centrality = centrality.copy()
        for node in G:
            centrality[node] = sum(prev_centrality[neighbor] for neighbor in in_neighbors(node))

        norm = np.sqrt(sum(value ** 2 for value in centrality.values()))
        if norm == 0:
            return centrality
        centrality = {node: value / norm for node, value in centrality.items()}

        max_diff = max(abs(centrality[node] - prev_centrality[node]) for node in G)
        if max_diff < tol:
            break

    return centrality


# 3. Subgraph Centrality
def my_subgraph_cent(G, normalized=False):
    """
    Compute the subgraph centrality for every node in a SIMPLE undirected graph.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be a networkx graph")
    if G.number_of_nodes() == 0:
        raise ValueError("K_0 has an undefined subgraph centrality")
    if G.is_directed():
        raise nx.NetworkXError("Graph must be Undirected")
    if nx.number_of_selfloops(G) > 0 or G.is_multigraph():
        raise nx.NetworkXError("Graph must be SIMPLE")

    nodelist = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodelist, weight=None)

    eigenvals, eigenvecs = np.linalg.eigh(A)

    if normalized:
        spec_rad = eigenvals[-1]
        shifted = np.exp(eigenvals - spec_rad)
        if not np.all(np.isfinite(shifted)):
            raise ValueError("Numerical overflow while normalizing")
        sc_vals = (eigenvecs ** 2) @ shifted
    else:
        sc_vals = (eigenvecs ** 2) @ np.exp(eigenvals)

    return {node: float(val) for node, val in zip(nodelist, sc_vals)}


# 4. PageRank Centrality
def compute_pagerank(adjacency_matrix, alpha=0.85, max_iter=100, tol=1.0e-6):
    """
    Computes PageRank centrality from scratch using power iteration.
    """
    M = np.array(adjacency_matrix, dtype=float)
    N = M.shape[0]

    out_degree = M.sum(axis=0)
    for j in range(N):
        if out_degree[j] == 0:
            M[:, j] = 1.0 / N
        else:
            M[:, j] /= out_degree[j]

    r = np.ones(N) / N
    teleport = np.ones(N) / N

    for _ in range(max_iter):
        r_prev = r.copy()
        r = alpha * np.dot(M, r_prev) + (1 - alpha) * teleport
        if np.sum(np.abs(r - r_prev)) < tol:
            break

    return r


# 5. Group Closeness Centrality
def group_closeness_centrality(graph, S):
    """
    Compute group closeness centrality from scratch using BFS.
    """
    # Accommodate NetworkX graphs directly or fallback to dictionary representation
    adj_list = graph.adj if hasattr(graph, 'adj') else graph
    V = set(adj_list.keys())
    S = set(S)
    V_minus_S = V - S

    if not V_minus_S:
        return 0.0

    distances = {node: -1 for node in V}
    queue = deque()

    for node in S:
        if node in adj_list:
            distances[node] = 0
            queue.append(node)

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        for neighbor in adj_list.get(current, []):
            if distances[neighbor] == -1:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

    total_distance = 0
    for node in V_minus_S:
        if distances[node] != -1:
            total_distance += distances[node]

    if total_distance == 0:
        return 0.0

    return len(V_minus_S) / total_distance


# 6. Centrality Visual Comparison Utility
def centrality_comparisons(G, pos, centralities):
    """
    Compare and plot different centrality measures for a graph dynamically.
    """
    num_plots = len(centralities)
    fig, ax = plt.subplots(1, num_plots, figsize=(4 * num_plots, 3.5), dpi=200, squeeze=False)

    col_map = {0: plt.cm.Oranges, 1: plt.cm.Blues, 2: plt.cm.Greens, 3: plt.cm.Purples}

    for i, (label, centrality_func) in enumerate(centralities.items()):
        cents = dict(centrality_func(G))
        # Ensure array order exactly tracks node positions
        cent_vals = np.array([cents[node] for node in G.nodes()])

        val_range = max(cent_vals) - min(cent_vals)
        cent_vals_minmax = (cent_vals - min(cent_vals)) / val_range if val_range > 0 else np.zeros_like(cent_vals)

        ncs = col_map[i % 4](cent_vals_minmax)
        nx.draw(G, pos, node_size=150, node_color=ncs, edgecolors='.2',
                with_labels=True, font_size='x-small', edge_color='.6', ax=ax[0, i])
        ax[0, i].set_title("Colored by\n" + label)

    os.makedirs('images/pngs', exist_ok=True)
    os.makedirs('images/pdfs', exist_ok=True)

    plt.savefig('images/pngs/centrality_compare_networkx.png', dpi=425, bbox_inches='tight')
    plt.savefig('images/pdfs/centrality_compare_networkx.pdf', bbox_inches='tight')
    plt.show()
