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

def levenschtein_dist(phrase1, phrase2):
    phrase1 = phrase1.translate(str.maketrans('', '', string.punctuation))
    phrase2 = phrase2.translate(str.maketrans('', '', string.punctuation))

    # Split the phrases into words
    words1 = phrase1.split()
    words2 = phrase2.split()
    words1 = word_tokenize(phrase1.lower())
    words2 = word_tokenize(phrase2.lower())
    dist = pylev.levenschtein(words1, words2)
    return dist



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
def make_graph_df_v2(df, k,d):

    G = nx.DiGraph()
    G.add_nodes_from(df['phrases'])

    phrases = df['phrases']
    lengths = df['len'].values
    for i,phr1 in enumerate(phrases):
        length_fr1 = lengths[i]
        length_filter = df['len']>length_fr1
        df_filter = df[length_filter].copy()
        
        df_filter['comp'] = df_filter['phrases'].apply(lambda x: count_consecutive_words(phr1, x))
        df_filter['dist'] = df_filter['phrases'].apply(lambda x: levenschtein_dist(phr1, x))

        filter_words = df_filter['comp'] >= k
        filter_dist = df_filter['dist'] <= d

        add_nodes, add_frec =  df_filter[filter_words | filter_dist][['phrases', 'frec']].values.T.tolist()

        for nodes, frecs in zip(add_nodes, add_frec):
            G.add_edge(phr1, nodes, weight = frecs)

    return G


def make_graph_df_v3(df, k,d):

    G = nx.DiGraph()
    G.add_nodes_from(df['phrases'])


    #for i,phr1 in enumerate(phrases):
    for i, row in df.iterrows():
        phr1 = row['phrases']
        length_fr1 = row['len']
        length_filter = df['len']>length_fr1
        df_filter = df[length_filter].copy()
        
        df_filter['comp'] = df_filter['phrases'].apply(lambda x: count_consecutive_words(phr1, x))
        df_filter['dist'] = df_filter['phrases'].apply(lambda x: levenschtein_dist(phr1, x))

        filter_words = df_filter['comp'] >= k
        filter_dist = df_filter['dist'] <= d

        add_nodes, add_frec =  df_filter[filter_words | filter_dist][['phrases', 'frec']].values.T.tolist()

        for nodes, frecs in zip(add_nodes, add_frec):
            G.add_edge(phr1, nodes, weight = frecs)

    return G

def tokenize_count(phrase):
    phrase = phrase.translate(str.maketrans('', '', string.punctuation))
    phrase = word_tokenize(phrase.lower())

    return len(phrase)

#path = 'd:/Facultad/Tesis/'
path = 'c:/Facultad/Tesis/'
phrases = extrac_phrases(path, 1000)


duration = 1000
freq = 440
winsound.Beep(freq, duration)

#%%
phrases_ones = list(set(phrases))
phdf = pd.DataFrame({'phrases': phrases_ones})
phdf['len'] = phdf['phrases'].apply(tokenize_count)
frec = [0]*len(phrases_ones)
for i, phr in enumerate(phrases_ones):
    frec[i] = phrases.count(phr)
phdf['frec'] = frec
#%%
#phdf.to_csv(path+'all_data.csv')
# with open('phrases.txt', 'w') as f:
#     for phr in phrases:
#         f.write(f"{phr}\n")

with open(path+'phrases_list.txt', 'wb') as f:
    pickle.dump(phrases, f) 

#%% Histograma de pesos
plt.hist(phdf['frec'], bins= np.linspace(0, 20000, 100))
#plt.xlim([0, 2500])
plt.yscale('log')
plt.ylabel('cantidad de apariencias')
plt.xlabel('pesos')


#%%

print(len(phrases))
t0 = time.time()
grafo = make_graph_df_v3(phdf, k = 10, d = 1)
tf = time.time()
print(tf-t0)


df_to_compare = pd.DataFrame(columns=['frases', 'cantidad'])
print(df_to_compare)
components = list(nx.weakly_connected_components(grafo)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado
for i,componente in enumerate(components):
    if len(componente) > 5:
        print(len(componente), list(componente))
        df_to_compare.loc[i] = [list(componente), len(componente)]

print(df_to_compare)

# plt.figure()
# nx.draw_kamada_kawai(grafo, node_size = 10)
# plt.show()


duration = 1000
freq = 440
winsound.Beep(freq, duration)
#%%

list(df_to_compare['frases'])

with open(path+'frases_to_compare.txt', 'wb') as f:
    pickle.dump(list(df_to_compare['frases']), f) 

with open(path+'cantidad_to_compare.txt', 'wb') as f:
    pickle.dump(list(df_to_compare['cantidad']), f) 


#df_to_compare.to_csv(path+'df_to_compare.csv')








#%% TESTTT
ar = np.arange(1,15)
edges = [(1,4),(1,5),(4,8),(5,8),(5,9),(9,13), (2,5),(5,10),(10,14),(2,6),(6,10),(6,11),(11,14),(3,6),(3,7),(7,11), (8,13)]
weights = np.ones(len(edges))
weights[1] = 3

g = nx.DiGraph()
g.add_edges_from(edges,weight = weights)
# plt.figure()
# nx.draw(g,with_labels = True)
# plt.show()

out = dict(g.out_degree())
in_ = dict(g.in_degree())

nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]

pto_13 = []
for n_out in nodes_cero_out:
    for path in nx.all_simple_paths(g, nodes_cero_in_[0], n_out):
        pto_13.append(path)

caminos = np.array(pto_13)

#%%

if len(np.unique(caminos[:,-1])) != 1:
    for i in range(len(caminos[0])):

for i in set(caminos[:,1]):
#    print(nx.ancestors(g, i))
    if len(nx.ancestors(g, i)) != 1:
        for j in nx.ancestors(g, i):
            print(g.get_edge_data(j,i))
            print(j,i)
#np.array(list(dict(out).values()))
# comp_t = components[14]
# print(comp_t)
# out = grafo.out_degree(comp_t)
# for o in out:
#     print(o[1])
# p = 0
# for path in nx.all_simple_paths(grafo, list(comp_t)[-3], list(comp_t)[3]):
#     print('path', path)
#     p += 1
# print(p)

# %%
