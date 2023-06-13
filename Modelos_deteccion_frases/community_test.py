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
#from community import community_louvain as com
path = 'd:/Facultad/Tesis/'

# path = 'd:/Git-proyects/Tesis/Modelos_deteccion_frases/'
def cluster_to_dict(cluster, g):
    dic = {}
    for i, c in enumerate(sorted(list(cluster), key = len, reverse = True)):
        for n in c:
            dic[g.vs[n]['_nx_name']] = i
    return dic

#grafo = pickle.load(open(path+'grafos/grafo_id150000_archivos.pickle', 'rb'))
path1 = 'd:/Git_Proyects/Tesis/Modelos_deteccion_frases/grafos/'


grafo = pickle.load(open(path1+'grafo_1000_k_3_weighted_v2.pickle', 'rb'))

components = list(nx.weakly_connected_components(grafo))


#%%
len_list = []
for i in components:
    len_list.append(len(i))
print(len(np.where(np.array(len_list) == 1)[0]))
plt.plot(len_list, '.')
plt.yscale('log')
#%%
for i in list(np.where(np.array(len_list) == 1))[0]:
    print(components[i])

lengths = []
for i in list(np.where(np.array(len_list) == 1))[0]:
    lengths.append(len(max(components[i]).split(" ")))

plt.plot(lengths, '.')
#%%
sub_graf = grafo.subgraph(components[0])
G_ig = ig.Graph.from_networkx(sub_graf) 
com_ip = G_ig.community_infomap()

dic_ip = cluster_to_dict(com_ip, G_ig)
#%%
[key for key, val in dic_ip.items() if val == 10]
#%%
positions = np.where(np.array(lengths) ==7)[0][0]
pos_c = np.where(np.array(len_list) == 1)[0][positions]
print(components[pos_c])


#%%
for i, comp in enumerate(components[0:1]):
    if len(comp) > 15:
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

#%%
c = 0
for comp in components:
    if len(comp)>1:
        c+=1

#%% Testing 4 algorithms of communits at the same time no funca

for j, comp in enumerate(components):
    if len(comp) > 25:
        sub_graf = grafo.subgraph(comp)
        sub_graf = sub_graf.to_undirected()
        G_ig = ig.Graph.from_networkx(sub_graf)

        pos = nx.layout.circular_layout(sub_graf)


        com_bt = G_ig.community_edge_betweenness(clusters = None, directed = False, weights = None)
        com_fg = G_ig.community_fastgreedy(weights = None)
        com_ip = G_ig.community_infomap()
        com_lv = com.best_partition(sub_graf)

        dic_com_bt = cluster_to_dict(com_bt.as_clustering(), G_ig)
        dic_com_fg = cluster_to_dict(com_fg.as_clustering(), G_ig)
        dic_com_ip = cluster_to_dict(com_ip, G_ig)
        particiones = [dic_com_bt, dic_com_fg, dic_com_ip, com_lv]
        part_name = ['Betweenness', 'Fast-Gready', ' InfoMap', 'Louvain']
        fig, axs = plt.subplots(2, 2, figsize = (15, 15))

        for i, ax in enumerate(fig.axes):
            ax.set_title(part_name[i])
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