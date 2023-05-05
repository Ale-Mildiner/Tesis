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
import pandas as pd

def count_consecutive_words(phrase1, phrase2):
    phrase1 = phrase1.translate(str.maketrans('', '', string.punctuation))
    phrase2 = phrase2.translate(str.maketrans('', '', string.punctuation))

    # Split the phrases into words
    words1 = phrase1.split()
    words2 = phrase2.split()
    words1 = word_tokenize(phrase1.lower())
    words2 = word_tokenize(phrase2.lower())


    # Initialize a counter and maximum value to zero
    count = 0
    max_count = 0

    # Loop through each word in the first phrase
    for i in range(len(words1)):
        # Check if the current word is in the second phrase
        if words1[i] in words2:
            # Find the index of the word in the second phrase
            j = words2.index(words1[i])
            # Loop through the remaining words in both phrases and count the consecutive matching words
            while i < len(words1) and j < len(words2) and words1[i] == words2[j]:
                count += 1
                i += 1
                j += 1
            # Update the maximum count value
            max_count = max(max_count, count)
            # Reset the counter to zero
            count = 0
    # Return the maximum count of consecutive words
    return max_count


def tokenize(phrase):
    phrase = phrase.translate(str.maketrans('', '', string.punctuation))
    phrase = word_tokenize(phrase.lower())

    return phrase


def make_graph_t(variables, k, d):
    '''
    variables: lista con los str para aramr la red
    k: parametro de la cantidad de palabras consevutivas para armar un link (segun paper k = 10)
    d: parametro delta de direct editing distance (segun paper d =< 1)
    '''
    variables_set = list(set(variables)) # si hago esto se reducen muchisimo los tiempos
    G = nx.DiGraph()
    G.add_nodes_from(variables_set)
    edges = []
    for i, phrase1 in enumerate(variables_set):
        for j, phrase2 in enumerate(variables_set):
            #if i != j: # ignoramos la diagonal
            if phrase1 != phrase2: #ignoramos las que son exactamente iguales
                phr1 = tokenize(phrase1)
                phr2 = tokenize(phrase2)
                lev_dist = pylev.levenschtein(phr1, phr2)     # Distancia de levenschtein, creo que es la que usan en el paper
                frec = variables.count(phrase1)

                if (count_consecutive_words(phrase1,phrase2) >= k  or lev_dist <= d) and len(phr1) < len(phr2):
                    G.add_edge(phrase1, phrase2, weight = frec)
 #                   edges.append((phrase1, phrase2))

    return G

def extrac_phrases(path, num):
    ids = np.arange(10, num)
    phrases = []
    for id in ids:
        try:
            data = pd.read_csv(path + 'Data_disgregada/Lkvec_id'+str(id)+'.csv')  
            phrases = phrases + list(data['phrase'])
        except:
            pass

    return phrases
#%% Prueba usando DataFrames
path = 'd:/Facultad/Tesis/'
phrases = extrac_phrases(path, 1000)


def tokenize_count(phrase):
    phrase = phrase.translate(str.maketrans('', '', string.punctuation))
    phrase = word_tokenize(phrase.lower())

    return len(phrase)


phrases_ones = list(set(phrases))
phdf = pd.DataFrame({'phrases': phrases_ones})
phdf['len'] = phdf['phrases'].apply(tokenize_count)
frec = [0]*len(phrases_ones)
for i, phr in enumerate(phrases_ones):
    frec[i] = phrases.count(phr)
phdf['frec'] = frec


def make_graph_df(df, k,d):

    G = nx.DiGraph()
    G.add_nodes_from(df['phrases'])
    for phr1 in df['phrases']:
        length_fr1 = int(df.loc[df['phrases'] == phr1]['len'])
        frec_fr1 = int(df.loc[df['phrases'] == phr1]['frec'])
        length_filter = df['len']>length_fr1
        df_filter = df[length_filter]
        for phr2 in df_filter['phrases']:
            phrese1 = tokenize(phr1)
            phrese2 = tokenize(phr2)
            lev_dist = pylev.levenschtein(phrese1, phrese2) 
            if (count_consecutive_words(phr1,phr2) >= k  or lev_dist <= d):
                    G.add_edge(phr1, phr2, weight = frec_fr1)

    return G

print(len(phrases))
t0 = time.time()
grafo = make_graph_df(phdf, k = 10, d = 1)
tf = time.time()
print(tf-t0)

components = list(nx.weakly_connected_components(grafo)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado
for componente in components:
    if len(componente) > 5:
        print(len(componente), list(componente))

#%%
path = 'd:/Facultad/Tesis/'
phrases = extrac_phrases(path, 1000)

print(len(phrases))
t0 = time.time()
grafo = make_graph_t(phrases, k = 10, d = 1)
tf = time.time()
print(tf-t0)


duration = 1000
freq = 440
winsound.Beep(freq, duration)
#%%
plt.figure()
nx.draw_kamada_kawai(grafo, node_size = 10)
plt.show()
print(len(grafo.edges()))

components = list(nx.weakly_connected_components(grafo)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado
for componente in components:
    if len(componente) > 5:
        print(len(componente), list(componente))
winsound.Beep(freq, duration)

#%% Pureba para eliminacion de enlances

#print(grafo.edges(nodes[2]))
# for node in nodes:
#     n_comp = 0
#     for comp in components:
#         if nodes[2] in comp:
#             n_comp += 1

def problematic_nodes(red):
    components = list(nx.weakly_connected_components(red)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado


    nodes = list(red.nodes())
    problematic_nodes = []
    for node in nodes:
        num_components = sum([node in component for component in components])
        if num_components > 1:
            problematic_nodes.append(node)

    return problematic_nodes

def edges_to_remove(red):
    '''
    Hay que iterar sobre todos los nodos (remover nodes[2]) y tener un criterio de remosion de 
    enlaces.
    '''
    components = list(nx.weakly_connected_components(red)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado

    comp_values = dict.fromkeys(components, 0) # creo un diccionario con las componentes y como valores va a tener la cantidad de apariciones
    for nodes_edges in grafo.edges(nodes[2]): # itero por los nodos conectados a un determinado nodo
        for comp in components:
            if nodes_edges in comp:
                comp_values[comp] += 1 
    
    for comp, appearances in comp_values.items():
        if appearances > 0:
            print(appearances) # tengo que probar bien pero no tengo datos para esto 
            #nx.remove_edge(node1, node2)
    return red  
#%% TEST de parametros
def dist_frases(componentes, min_conect):
    values = []
    for comp in componentes:
        if len(comp) > min_conect:
            values.append(len(comp))
    return values

ks = [8,10,12] # cantidad de palabras consecutivas
ds = [1,2,3] # a medida que aumenta le permitimos qumente mas la distancia de edicion
path = 'd:/Facultad/Tesis/'
phrases = extrac_phrases(path, 1000)

dist_k_variando = []
for k in ks:
    g_k = make_graph_t(phrases, k, 1)
    components = list(nx.weakly_connected_components(g_k)) 
    dist_k_variando.append(dist_frases(components, 2))
dist_d_variando = []
for d in ds:
    g_d = make_graph_t(phrases, 10, d)
    components = list(nx.weakly_connected_components(g_d)) 
    dist_d_variando.append(dist_frases(components, 2))

duration = 1000
freq = 440
winsound.Beep(freq, duration)
#%%
plt.figure()
plt.title('Variando k')
for i in range(len(dist_k_variando)):
    plt.hist(dist_k_variando[i], label = ks[i], alpha = 0.7)
plt.legend()
plt.show()

plt.figure()
plt.title(r'Variando $\delta$')
for i in range(len(dist_d_variando)):
    plt.hist(dist_d_variando[i], label = ds[i], alpha = 0.4)
plt.legend()
plt.show()