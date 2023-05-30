#%%
from nltk import word_tokenize
import string
import pylev
import networkx as nx
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import pickle
from collections import OrderedDict
import os 

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
def extrac_phrases_weighted(path1, path2, number_files):
    sorted_files = pickle.load(open(path1+'sorted_files.pickle', 'rb'))

    phrases = []
    i= 0
    while i < number_files:
        file_path = os.path.join(path2, sorted_files[i][0])
        data = pd.read_csv(file_path)  
        phrases = phrases + list(data['phrase'])
        i += 1

    phrases_ones = list(OrderedDict.fromkeys(phrases))
    
    return phrases, phrases_ones

path1 = 'd:/Git_Proyects/Tesis/Modelos_deteccion_frases/'
path2 = 'd:/Facultad/Tesis/Data_disgregada'
#%%
# path = 'd:/Facultad/Tesis/'
# phrases = extrac_phrases(path, 1000)           

# phrases_ones = list(set(phrases))
phrases, phrases_ones = extrac_phrases_weighted(path1, path2, 100)
phdf = pd.DataFrame({'phrases': phrases_ones})
phdf['len'] = phdf['phrases'].apply(tokenize_count)
frec = [0]*len(phrases_ones)
for i, phr in enumerate(phrases_ones):
    frec[i] = phrases.count(phr)
phdf['frec'] = frec

print(len(phrases))
t0 = time.time()
grafo = make_graph_df_v3(phdf, k = 10, d = 1)
tf = time.time()
print(tf-t0)


#pickle.dump(grafo, open('grafo_2300_archivos.pickle', 'wb'))

# #%%
# for file in sorted_files:
#     file_path = os.path.join(path, file[0])
#     print(file_path)
#     data = pd.read_csv(file_path)  
#     phrases = phrases + list(data['phrase'])





# #%%
# import os
# import operator
# import winsound
# def file_size(path):
#     return os.path.getsize(path)

# path = 'd:/Facultad/Tesis/Data_disgregada/'
# files_name = os.listdir(path)
# file_sizes = [(file, file_size(os.path.join(path, file))) for file in files_name]
# sorted_files = sorted(file_sizes, key=operator.itemgetter(1), reverse=True)

# duration = 1000
# freq = 440
# winsound.Beep(freq, duration)

# #%%
# for file, size in sorted_files:
#     file_path = os.path.join(path, file)
#     # Realiza las operaciones necesarias con el archivo
#     print(file_path, size)  # Ejemplo: Imprimir la ruta y el tamaño del archiv


