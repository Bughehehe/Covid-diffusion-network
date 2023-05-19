import networkx as nx
import matplotlib.pyplot as plt

G = nx.erdos_renyi_graph(3, 2)

nx.draw(G)
plt.figure(2)
plt.hist(nx.centrality.closeness_centrality(G).values())
plt.show()