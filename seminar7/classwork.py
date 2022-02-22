import networkx as nx

G = nx.DiGraph()
G.add_node("Hello")
G.add_node("World")
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_edge("Hello", "World")
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 4)
G.add_edge(1, 1)
print(G.nodes)
print(G.edges)
print(G.out_edges(1))
print(G.out_degree(1))
G.nodes[3]["weight"] = 17
print(G.nodes[3])
G.edges[(1, 2)]["color"] = "red"
print(G.edges[(1, 2)])

nx.draw(G, with_labels=True)
import matplotlib.pyplot as plt
plt.savefig("graph.png")

nx.write_graphml(G, "graph.graphml")
