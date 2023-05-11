#%%
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

nodes = np.arange(0, 20)
frecs = np.zeros(len(nodes))

for i in range(len(nodes)):
    frecs[i] = random.randint(1,100)
#%%
g = nx.DiGraph()
g.add_nodes_from(nodes)

prob = 0.2
for ni in nodes:
    for j, nj in enumerate(nodes):
        ran = random.random()
        if ran < prob and nj>ni:
            g.add_edge(ni, nj, weight = frecs[j])

#%%
nx.draw_circular(g, with_labels = True)

def cant_caminos(g):
    out = dict(g.out_degree())
    in_ = dict(g.in_degree())

    nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
    nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]

    cant_path = 0
    for n_in in nodes_cero_in_:
        for n_out in nodes_cero_out:
            cant_path = cant_path+ len(list(nx.all_simple_paths(g, n_in, n_out)))

    return cant_path, nodes_cero_out, nodes_cero_in_

out = dict(g.out_degree())
in_ = dict(g.in_degree())
nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]
nodes_cero_in_

#n_in = nodes_cero_in_[0]

def cont_p_nin(n_in, n_cero_out):
    cont = 0
    for out in n_cero_out:
        cant_de_caminos = len(list(nx.all_simple_paths(g, n_in, out)))
        if cant_de_caminos > 0:
            cont += 1
    return cont

#%%
df = pd.DataFrame(columns=['enlaces', 'pesos'])
for i, edge in enumerate(g.edges()):
    df.loc[i] = edge, g.get_edge_data(edge[0], edge[1]).get('weight')
df.sort_values(by = ['pesos'], ascending=True)

enlces_en_orden = list(df.sort_values(by=['pesos'], ascending=True)['enlaces'])
pesos = list(df.sort_values(by=['pesos'], ascending=True)['pesos'])
#print(enlces_en_orden)
out = dict(g.out_degree())
in_ = dict(g.in_degree())
nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]

for nodes_in in nodes_cero_in_:
    cant_in_out_prev = cont_p_nin(nodes_in, nodes_cero_out)
    #print(cant_in_out_prev)
    i = 0
    while (cant_in_out_prev > 1) and (i <len(enlces_en_orden)):
        g.remove_edge(enlces_en_orden[i][0], enlces_en_orden[i][1])
        cant_in_out = cont_p_nin(nodes_in, nodes_cero_out)
        if not cant_in_out_prev > cant_in_out:
            g.add_edge(enlces_en_orden[i][0], enlces_en_orden[i][1], weight = pesos[i])
        else:
            enlces_en_orden.pop(i)
        cant_in_out_prev = cant_in_out
        i += 1
    print(nodes_in, cant_in_out_prev)
#%%
nx.draw_circular(g, with_labels = True)



#%% ESTO FUNCAAAAAAAAAAAAAAAAA!!!!!!!!!!!!!!!!!!!!!!!!!!
ar = np.arange(1,16)
edges = [(1,4),(1,5),(4,8),(5,8),(5,9),(9,13), (2,5),(5,10),(10,14),(2,6),(2,7),(6,10),(6,11),(10,14),(11,14),(3,6),(3,7),(7,11),(8,13),(7,15),(12,15)]
weights = np.ones(len(edges))
weights[1] = 3
weights[7] = 0.3
weights[-2]= 0.7
g = nx.DiGraph()
g.add_nodes_from(ar)
g.add_edges_from(edges,weight = weights)
nx.draw_circular(g, with_labels = True)


df = pd.DataFrame(columns=['enlaces', 'pesos'])
for i, edge in enumerate(edges):
    df.loc[i] = edge, g.get_edge_data(edge[0], edge[1]).get('weight')[i]
df.sort_values(by = ['pesos'], ascending=True)

enlces_en_orden = list(df.sort_values(by=['pesos'], ascending=True)['enlaces'])
pesos = list(df.sort_values(by=['pesos'], ascending=True)['pesos'])
print(enlces_en_orden)
out = dict(g.out_degree())
in_ = dict(g.in_degree())
nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]

for nodes_in in nodes_cero_in_:
    cant_in_out_prev = cont_p_nin(nodes_in, nodes_cero_out)
    print(cant_in_out_prev)
    i = 0
    while (cant_in_out_prev > 1) and (i <len(enlces_en_orden)):
        g.remove_edge(enlces_en_orden[i][0], enlces_en_orden[i][1])
        cant_in_out = cont_p_nin(nodes_in, nodes_cero_out)
        if  not cant_in_out_prev > cant_in_out:
            g.add_edge(enlces_en_orden[i][0], enlces_en_orden[i][1], weight = pesos[i])
        else:
            enlces_en_orden.pop(i)
        cant_in_out_prev = cant_in_out
        i += 1
#%%
nx.draw_circular(g, with_labels = True)


#%% Deprecatied
cant_path_prv, nodes_cero_out, nodes_cero_in = cant_caminos(g)
i = 1
while len(nodes_cero_out) > 1:
    g.remove_edge(enlces_en_orden[i][0], enlces_en_orden[i][1])
    cant_path, nodes_cero_out, nodes_cero_in = cant_caminos(g)
    if not cant_path < cant_path_prv:
        g.add_edge(enlces_en_orden[i][0], enlces_en_orden[i][1], weight = pesos[i])

    cant_path_prv = cant_path
    i = i+1
#%%
nx.draw_circular(g, with_labels = True)