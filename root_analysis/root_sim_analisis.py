#%%
import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')


path1 = 'd:/Facultad/Tesis/'

roots = pk.load(open(path1+'root_phrases_unique.pickle', 'rb'))
ids = pk.load(open(path1+'root_ids.pickle', 'rb'))
roots_emb = pk.load(open(path1+'root_emb.pickle', 'rb'))
pos_f = pk.load(open('filas_index_08.pickle', 'rb'))
pos_c = pk.load(open('columnas_index_08.pickle', 'rb'))



sim  = cosine_similarity(roots_emb[pos_c[0]].reshape(1,-1), roots_emb[pos_f[0]].reshape(1,-1))[0,0]
sim
val = 2
print(roots[pos_f[val]], 'ids ', ids[pos_f[val]])
print(roots[pos_c[val]], 'ids ', ids[pos_c[val]])

#%%
tuple_list = []
for i in range(len(pos_c)):
    t = (pos_f[i],pos_c[i])
    tupla_inversa = (t[1], t[0])
    if tupla_inversa not in tuple_list:
        tuple_list.append(t)
