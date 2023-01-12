import networkx as nx
import matplotlib.pyplot as plt

# Create a graph using the Graph class
G = nx.Graph()

# Add edges to the graph, with weights
G.add_edge(1, 2, weight=2.0)
G.add_edge(1, 3, weight=3.0)
G.add_edge(1, 4, weight=4.0)
G.add_edge(2, 3, weight=5.0)
G.add_edge(2, 5, weight=6.0)
G.add_edge(3, 4, weight=7.0)
G.add_edge(3, 6, weight=8.0)
G.add_edge(4, 5, weight=9.0)
G.add_edge(4, 6, weight=10.0)
G.add_edge(5, 6, weight=11.0)
# G = nx.complement(G)
# Set the layout of the graph
pos = nx.kamada_kawai_layout(G)
# pos = nx.planar_layout(G)

# Use the 'colors' dictionary to assign colors to the nodes
# colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple'}
# Colors = nx.greedy_color(G)

# Set the length of each edge to its weight
for u, v, weight in G.edges(data='weight'):
    G[u][v]['length'] = weight

# Use the draw() function to draw the graph
# nx.draw(G, pos=pos, with_labels=True, node_color=[colors[Colors[n]] for n in G.nodes()])
nx.draw(G, pos=pos, with_labels=True)
# Add the edge labels
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)

# Display the graph
plt.show()
