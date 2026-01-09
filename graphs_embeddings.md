# Graphs & Embeddings

## Graphs

Graphs generally have the structure $G = (V, E)$ where the graph $G$ has a set of edges $e \in E$ and vertices $v \in V$.

Graphs can be classified into two types based on the properties of their individual edges:
- **Simple Graphs** wherein an single edge $e=(u,v)$ can connect only two vertices $u$ and $v$. 
- **Hypergraphs** are a generalization of simple graphs differentiated by having edges that can join more than two vertices with each other $e=(v_1, v_2, \dots, v_n)$

Hypergraphs are remarkably exprressive in terms of being able to map relational properties between entities represented by vertices.
Consider any product as an edge $e$ capable of connecting multiple users $\{v_1, \dots, v_n\}$ based on whether they have purchased the item represented by $e$.
Similarly we can have other relational properties that can be expressed by such a model.

```python
import matplotlib.pyplot as plt
import numpy as np
import xgi

H = xgi.Hypergraph()
H.add_edges_from(
    [[1, 2, 3], [1, 2, 3, 4, 5, 6]]
)

xgi.draw(H, pos=pos, ax=ax, hull=True)

plt.show()
```
![Hypergraph containing the vertices 1, 2, 3, 4, 5, 6 and the hyperedges [1, 2, 3], [1, 2, 3, 4, 5, 6]](assets/hypergraph_H01.png)

Simple Graphs and Hypergraphs are interchangeable structures, we can *expand* a hypergraph into a simple graph and condense a simple graph into a hypergraph.
The process of converting a hypergraph into a simple graph is called *hypergraph expansion*.
There are primarily two types of hypergraph expansion processes:
- **Clique Expansion** - each each hyperedge is transformed into a clique such that every pair of nodes belonging to a hyperedge is always directly connected. This generates a lot of edges but helps in retaining relationships represented by the hyperedges.
- **Star Expansion** - Alternatively, a simpler strategy is to introduce new *virtual nodes* connecting all original nodes in a hyperedge. This aids in reducing the number of edges generated during the expansion process but still preserves hyperedge relationships.

<!--diagram of the two expansion techniques here-->

Graphs and the relations that they represent by themselves are of little value, we need to extract insights from these graphical structures in order to 




## Embeddings
