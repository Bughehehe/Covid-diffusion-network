import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

def SI(G,Nb_inf_init,HM, N, T):
    pos = nx.spring_layout(G)
    A = nx.to_numpy_array(G)
    # setting initial conditions
    sup = np.zeros(T)
    inf = np.zeros(T)
    nb_inf_t = np.zeros(T)
    inf[0] = Nb_inf_init
    sup[0] = N - Nb_inf_init
    nb_inf_t[0] = inf[0]
    for u in G.nodes():
        G.nodes[u]["state"] = 0
        G.nodes[u]["noeux_associes"] = [n for n in G.neighbors(u)]
    init = random.sample(list(G.nodes()), Nb_inf_init)
    for u in init:
        G.nodes[u]["state"] = 1

    for t in range(1,T):
        sup[t] = sup[t-1]
        inf[t] = inf[t-1]
        for u in G.nodes:
            # si infecté
            if G.nodes[u]["state"] == 0:
                nb_friend_infected = [G.nodes[n]["state"] == 1 for n in G.nodes[u]["noeux_associes"]].count(True)
                for n in G.nodes[u]["noeux_associes"]:
                    if G.nodes[n]["state"] == 1: # si voisin infecté
                        # with HM infect
                        if np.random.rand() < HM:
                            G.nodes[u]["state"] = 1
                            inf[t] += 1
                            sup[t] += -1
                            break
        nb_inf_t[t] = inf[t]-inf[t-1]
    return sup, inf, nb_inf_t

def SIR(G,Nb_inf_init,Gamma,HM, N, T):
    """ function that runs a simulation of an SIR model on a network.
    Args:
        Gamma(float): recovery rate
        Beta(float): infection probability
        Rho(float): initial fraction of infected individuals
        N(int): number of agents (nodes)
        T(int): number of time steps simulated
    """
    A = nx.to_numpy_array(G)
    # setting initial conditions
    s = np.zeros(T)
    inf = np.zeros(T)
    r = np.zeros(T)
    inf[0] = Nb_inf_init
    s[0] = N - Nb_inf_init
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.nodes[u]["state"] = 0
        G.nodes[u]["TimeInfected"] = 0
        G.nodes[u]["noeux_associes"] = [n for n in G.neighbors(u)]

    init = random.sample(list(G.nodes()), Nb_inf_init)
    for u in init:
        G.nodes[u]["state"] = 1
        G.nodes[u]["TimeInfected"] = 1
    # running simulation
    for t in range(1,T):
        s[t] = s[t-1]
        inf[t] = inf[t-1]
        r[t] = r[t-1]
        # Check which persons have recovered
        for u in G.nodes:
            # if infected
            if G.nodes[u]["state"] == 1:
                if G.nodes[u]["TimeInfected"] < Gamma:
                    G.nodes[u]["TimeInfected"] += 1
                else:
                    G.nodes[u]["state"] = 2 #"recovered"
                    r[t] += 1
                    inf[t] += -1
        # check contagion    
        for u in G.nodes:
            #if susceptible
            if G.nodes[u]["state"] == 0:
                nb_friend_infected = [G.nodes[n]["state"] == 1 for n in G.nodes[u]["noeux_associes"]].count(True)
                #print(nb_friend_infected)
                for n in G.nodes[u]["noeux_associes"]:
                    if G.nodes[n]["state"] == 1: # if friend is infected
                        # with HM infect
                        if np.random.rand() < HM:
                            G.nodes[u]["state"] = 1
                            inf[t] += 1
                            s[t] += -1
                            break
    
    return s, inf,r

np.random.seed(0)
# time of simulation
T = 100
# number of agents
N = 500
HM = 0.1
Gamma = 5
Nb_inf_init = 5
# mean degree of the networks
k = 20
# defining an erdos renyi network
G = nx.erdos_renyi_graph(N,k/N)
nx.draw(G)
plt.figure(2)
s_erdos, inf_erdos,r_erdos = SIR(G,Nb_inf_init,Gamma,HM, N, T)
plt.plot((100/N)*s_erdos, color='b',marker='+', label="Susceptible k=20")
plt.plot((100/N)*inf_erdos, color='r',marker='+', label="Infected p=0.04")
plt.plot((100/N)*r_erdos, color='g',marker='+', label="Recovered p=0.04")
plt.xlabel("time")
plt.ylabel("Percentage of population infected")
plt.legend()
plt.show()


# mean degree of the networks
k = 10
# defining an erdos renyi network
G = nx.erdos_renyi_graph(N,k/N)

s_erdos, inf_erdos,r_erdos = SIR(G,Nb_inf_init,Gamma,HM, N, T)
plt.plot((100/N)*s_erdos,color="b",marker='o',label="Susceptible k=10")
plt.plot((100/N)*inf_erdos,color="r",marker='o',label="Infected p=0.02")
plt.plot((100/N)*r_erdos,color='g',marker='o',label="Recovered p=0.02")

plt.xlabel("time")
plt.ylabel("Percentage of population infected")
plt.legend()
plt.show()









# T = 100
# N = 500 # size of population
# HM = 0.03
# Nb_inf_init = 1 #  Initial numbre of infected person
# # mean degree of the networks
# k = 20
# # defining an erdos renyi network
# G = nx.erdos_renyi_graph(N,k/N)

# s_erdos, inf_erdos,nb_inf_t = SI(G,Nb_inf_init,HM, N, T)
# plt.plot((100/N)*s_erdos, color='b',marker='+', label="Susceptible p=0.04")
# plt.plot((100/N)*inf_erdos, color='r',marker='o', label="Infected  p=0.04")
# plt.xlabel("time")
# plt.ylabel("Percentage of population infected")
# plt.legend()
# plt.show()
# # mean degree of the networks
# k = 5
# # defining an erdos renyi network
# G = nx.erdos_renyi_graph(N,k/N)

# s_erdos, inf_erdos,nb_inf_t = SI(G,Nb_inf_init,HM, N, T)
# plt.plot((100/N)*s_erdos,"b",marker='+', label="Susceptible p=0.01")
# plt.plot((100/N)*inf_erdos,"r",marker='o', label="Infected  p=0.01")

# plt.xlabel("time")
# plt.ylabel("Percentage of population infected")
# plt.legend()
# plt.show()