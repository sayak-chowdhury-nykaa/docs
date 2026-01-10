# Graphs & Embeddings

## Graphs:

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

Graphs and the relations that they represent by themselves are of little value, we need to extract insights from these structures in order to make predictions or understand how entities relate to or interact with each other.
For recommender systems specifically, one of the most important utilities of such structures is the capability to analyze how different or similar two or more entities are.

For example, in a graph $G=(V, E)$ recording the social interactions between people in a population how social two entities $u\in V$ and $v\in V$ are is determined by the degree of their separation.
Their *social*-ness is inversely proportional to the number of edges connecting them.
Similarly, for any graph encoding the relationship between different entities, how similar two entities are depends on both of their *context*. 

> How does one determine the context of a node in a graph?

The above question can be answered by exploring the following properties of nodes in a graph:
1. **Homophily**, which is a term borrowed from sociology which means that similar entities tends to associate more closely with each other. This is most closely associated with Breadth-First Search (BFS), that analyzes the immediate surroundings of a particular node.
2. **Structural Equivalence**, which analyzes the role of a node from a wider perspective. This is most closely associated with Depth-First Search (DFS) techniques that evaluate the structure of a graph.

### Determining Context in Graphs:
Within any graph consider a source $\mu$ and a sink $v$, the task is to determine how similar the two nodes are to each other.
This can be found out by computing the context of either node and seeing how much overlap they have, the larger the overlap of their individual contexts is, the more similar they are.
For both of the nodes, we calculate their neighbourhoods, which are sets of nodes that are in close proximity to the respective nodes. For a sampling strategy $S$, the corresponding neighbourhoods $N_S( )$ are:

$$N_S(\mu)\coloneqq \{ x_1, \dots, x_n \}$$
$$N_S(v) \coloneqq \{ y_1, \dots, y_m \}$$

These can be computed using either BFS-like or DFS-like algorithms that try to span the entire graph.
Once these sets of nodes have been determined, their intersection can be taken as a metric to determine how similar the source and sink are.
However, this metric is not sufficient for densely connected graphs and overall does not provide a strong idea of the nature of similarity shared by the source and sink.
In the worst case, for a densely connected graph, every node may become completely similar to every other node while consuming significant resources to compute this response.

An alternative method that can be used to compute the context of a node is by using sampling techniques to approximate the neighbourhood of the individual nodes.
This would consequently give us the approximate context of the nodes in the graph.
An example of such a sampling method can be having a fixed number of parameterized random walks to explore the context of a node.
This parameterization enables modulating the trade-offs between DFS and BFS by incorporating heuristics or paramaters that can be biased towards certain kinds of exploration.

For machine learning tasks, similarity is often computed conditioned on the embeddings of each entity.
more about this can be found in the following section.

## Embeddings
Embeddings (or embedding vectors) are essentially some $d$-dimensional vectors that represent encodings of some high-level features.
These embeddings are trained to retain the signals that can be used to compute how similar two entities are.
Thus, the overall task is to learn a function $f$ that can compress graph-level features into a single vector $\vec{v}\in \mathbb{R}^d$.
The dimensionality of the vectors $d$ is a architectural hyperparameter and often determines how dense the representation learnt is going to be (a $100D$ vector can retain more information rather than a $3D$ vector).

$$\mu \to f(\mu) \to \vec{\mu} \in \mathbb{R}^d$$
$$v \to f(v) \to \vec{v} \in \mathbb{R}^d$$

Here, all embeddings are assumed to be living on the same $\mathbb{R}^d$ *embedding space*.
If the function accurately maps the graph-level properties to the embedding space, entities with overlapping context in the graph should have correspondingly high degree of similarity of context in $\mathbb{R}^d$.
For vectors, this similarity of context can be computed very simply by taking the dot products of the vector representation of the two entities.

$$\vec{\mu}\cdot \vec{v} = f(\mu)\cdot f(v)$$
