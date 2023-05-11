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


#%%
path = 'd:/Facultad/Tesis/'
#path = 'c:/Facultad/Tesis/'
phrases = extrac_phrases(path, 5000)


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

# with open(path+'phrases_list.txt', 'wb') as f:
#     pickle.dump(phrases, f) 

# #%% Histograma de pesos
# plt.hist(phdf['frec'], bins= np.linspace(0, 20000, 100))
# #plt.xlim([0, 2500])
# plt.yscale('log')
# plt.ylabel('cantidad de apariencias')
# plt.xlabel('pesos')


#%%

print(len(phrases))
t0 = time.time()
grafo = make_graph_df_v3(phdf, k = 10, d = 1)
tf = time.time()
print(tf-t0)
winsound.Beep(freq, duration)

#%%
df_to_compare = pd.DataFrame(columns=['frases', 'cantidad'])
print(df_to_compare)
components = list(nx.weakly_connected_components(grafo)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado
for i,componente in enumerate(components):
    if len(componente) > 5:
        #print(len(componente), list(componente))
        df_to_compare.loc[i] = [list(componente), len(componente)]
        sub_graf = grafo.subgraph(componente)
        out = dict(sub_graf.out_degree())
        in_ = dict(sub_graf.in_degree())
        nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
        nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]
        print('nodos out 0', len(nodes_cero_out))
print(df_to_compare)

# plt.figure()
# nx.draw_kamada_kawai(grafo, node_size = 10)
# plt.show()


duration = 1000
freq = 440
winsound.Beep(freq, duration)















#%% TEST Removien edges

def cont_p_nin(g, n_in, n_cero_out):
    cont = 0
    for out in n_cero_out:
        cant_de_caminos = len(list(nx.all_simple_paths(g, n_in, out)))
        if cant_de_caminos > 0:
            cont += 1
    return cont

def remove_edges(component):
    '''
    componente: subgrafo
    '''
    edges = list(component.edges())
    df = pd.DataFrame(columns=['enlaces', 'pesos'])
    for i, edge in enumerate(edges):
        df.loc[i] = edge, component.get_edge_data(edge[0], edge[1]).get('weight')
    df.sort_values(by = ['pesos'], ascending=True)
    enlces_en_orden = list(df.sort_values(by=['pesos'], ascending=True)['enlaces'])
    pesos = list(df.sort_values(by=['pesos'], ascending=True)['pesos'])
    out = dict(component.out_degree())
    in_ = dict(component.in_degree())
    nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
    nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]

    print(len(nodes_cero_in_), len(nodes_cero_out))
    for nodes_in in nodes_cero_in_:
        cant_in_out_prev = cont_p_nin(component, nodes_cero_in_, nodes_cero_out)
        i = 0
        if cant_in_out_prev <= 1:
            print('culo')
        while (cant_in_out_prev > 1) and (i <len(enlces_en_orden)):
            component.remove_edge(enlces_en_orden[i][0], enlces_en_orden[i][1])
            cant_in_out = cont_p_nin(component, nodes_in, nodes_cero_out)
            if  not cant_in_out_prev > cant_in_out:
                component.add_edge(enlces_en_orden[i][0], enlces_en_orden[i][1], weight = pesos[i])
            else:
                enlces_en_orden.pop(i)
            cant_in_out_prev = cant_in_out
            i += 1
    return component



df_to_compare = pd.DataFrame(columns=['frases', 'cantidad'])
#print(df_to_compare)
components = list(nx.weakly_connected_components(grafo)) #se puede usar weakly or strongly connectd, weakly coincide con un grafo no direccionado
for i,componente in enumerate(components):
    if len(componente) > 5:
        #print(len(componente), list(componente))
        df_to_compare.loc[i] = [list(componente), len(componente)]

        sub_graf = grafo.subgraph(componente)
        out = dict(sub_graf.out_degree())
        in_ = dict(sub_graf.in_degree())
        nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
        nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]
        print('nodos out', len(nodes_cero_out))
        try:
            #print('en try', sub_graf.nodes())
            if len(nodes_cero_out)>1:
                
                comp_divide = remove_edges(sub_graf)
                comps_div = list(nx.weakly_connected_components(comp_divide))
                print('div', len(comp_divide))
                
                for j, compss in enumerate(comps_div):
                    out = dict(compss.out_degree())
                    in_ = dict(compss.in_degree())
                    nodes_cero_out = [clave for clave, valor in out.items() if valor == 0]
                    nodes_cero_in_ = [clave for clave, valor in in_.items() if valor == 0]
                    #print('nodos_out_dividio', j, len(nodes_cero_out))

            else:
                pass
        except:
            pass
            #print('en except', sub_graf.nodes())
#print(df_to_compare)

# plt.figure()
# nx.draw_kamada_kawai(grafo, node_size = 10)
# plt.show()


duration = 1000
freq = 440
winsound.Beep(freq, duration)













#%% Save files

# list(df_to_compare['frases'])

# with open(path+'frases_to_compare.txt', 'wb') as f:
#     pickle.dump(list(df_to_compare['frases']), f) 

# with open(path+'cantidad_to_compare.txt', 'wb') as f:
#     pickle.dump(list(df_to_compare['cantidad']), f) 


#df_to_compare.to_csv(path+'df_to_compare.csv')








