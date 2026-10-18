# Network Centrality Measures From Scratch

A lightweight Python implementation of primary structural graph centrality algorithms from scratch.

##Local Package Installation
```bash
%pip install -e ./mypackage
```

## Comprehensive Metric Catalog

| Function Module | Applied Concept | Implementation Core |
| :--- | :--- | :--- |
| `closeness_centrality_from_scratch` | Shortest path reciprocal geodesic distance. | Single/Multi-component scaling adaptation. |
| `eigenvector_centrality_from_scratch` | Adjacency relative influence convergence vectors. | Power Iteration sequence tracking. |
| `my_subgraph_cent` | Weighted sum of self-returning closed walks. | Symmetric matrix Spectral Decomposition. |
| `compute_pagerank` | Teleportation-damped random surfer vectors. | L1-Norm converging stochastic power loops. |
| `group_closeness_centrality` | Combined distance metric from target sets to components. | Multi-source Breadth-First-Search (BFS). |
| `centrality_comparisons` | Multi-axes normalized coloring grid generator. | Visual Matplotlib rendering suite. |

## Live Evaluation Pipeline

```python
import networkx as nx
from mypackage import my_subgraph_cent, closeness_centrality_from_scratch, centrality_comparisons

# Instantiate the standard Karate Club Graph
G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42)

# Collate custom tracking pipelines
metrics = {
    "Closeness": closeness_centrality_from_scratch,
    "Subgraph Cent": lambda g: my_subgraph_cent(g, normalized=True)
}

# Render across analytical graphs
centrality_comparisons(G, pos, metrics)
```
![Dancing Character](skellybro.gif)
