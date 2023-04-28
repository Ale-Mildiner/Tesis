#%%
from nltk import word_tokenize
import string
import pylev
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

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
#%%
path = 'd:/Facultad/Tesis/'
phrases = extrac_phrases(path, 1000)

print(len(phrases))
t0 = time.time()
grafo = make_graph_t(phrases, k = 10, d = 1)
tf = time.time()
print(tf-t0)

#%%
plt.figure()
nx.draw_kamada_kawai(grafo, node_size = 10)
plt.show()
print(len(grafo.edges()))

components = list(nx.weakly_connected_components(grafo)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado
for componente in components:
    if len(componente) > 5:
        print(len(componente), list(componente))

#%% TEST de parametros
import winsound
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