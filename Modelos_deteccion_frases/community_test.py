#%%
from nltk import word_tokenize
import string
import pylev
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import winsound
import pickle
import igraph as ig
from community import community_louvain as com

# path = 'd:/Git-proyects/Tesis/Modelos_deteccion_frases/'
def cluster_to_dict(cluster, g):
    dic = {}
    for i, c in enumerate(sorted(list(cluster), key = len, reverse = True)):
        for n in c:
            dic[g.vs[n]['_nx_name']] = i
    return dic

grafo = pickle.load(open('grafo_300_archivos.pickle', 'rb'))
components = list(nx.weakly_connected_components(grafo))

for i, comp in enumerate(components):
    if len(comp) > 10:
        sub_graf = grafo.subgraph(comp)
        G_ig = ig.Graph.from_networkx(sub_graf)
        com_ip = G_ig.community_infomap()
        dic_ip = cluster_to_dict(com_ip, G_ig)
        pos = nx.layout.circular_layout(sub_graf)

        plt.figure()
        plt.title('comunidad'+ str(i))
        nx.draw_networkx_nodes(sub_graf, pos = pos, 
                node_color = [plt.get_cmap('Set1')(dic_ip[v]) for v in sub_graf.nodes()],
                node_size =100)
        nx.draw_networkx_edges(sub_graf,
                               pos = pos)
        plt.axis('off')
        plt.show()
        #print(com_ip)

#%% Testing 4 algorithms of communits at the same time

for j, comp in enumerate(components):
    if len(comp) > 10:
        sub_graf = grafo.subgraph(comp)
        G_ig = ig.Graph.from_networkx(sub_graf)

        pos = nx.layout.circular_layout(sub_graf)


        com_bt = G_ig.community_edge_betweenness(clusters = None, directed = False, weights = None)
        #com_fg = G_ig.community_fastgreedy(weights = None)
        com_ip = G_ig.community_infomap()
        #com_lv = com.best_partition(sub_graf)

        dic_com_bt = cluster_to_dict(com_bt.as_clustering(), G_ig)
        #dic_com_fg = cluster_to_dict(com_fg.as_clustering(), G_ig)
        dic_com_ip = cluster_to_dict(com_ip, G_ig)
        particiones = [dic_com_bt, dic_com_ip]
        
        fig, axs = plt.subplots(1, 2, figsize = (15, 15))

        for i, ax in enumerate(fig.axes):
            nx.draw_networkx_nodes(sub_graf,
                                pos = pos,
                                node_color = [plt.get_cmap('Set1')(particiones[i][v]) for v in sub_graf.nodes()],
                                ax = ax,
                                node_size = 100)
            nx.draw_networkx_edges(sub_graf,
                                pos = pos,
                                ax = ax,
                                alpha = .7)
            ax.axis('off')

        plt.show()