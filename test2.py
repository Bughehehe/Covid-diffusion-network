import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

G = nx.barabasi_albert_graph(20,5)

# nx.draw_spring(G)
nx.draw(G)
# nx.draw_planar(G)
plt.figure(2)
plt.hist(nx.centrality.closeness_centrality(G).values())
plt.show()