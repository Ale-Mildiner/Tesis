import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from numpy.linalg import norm

def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

def comparation(Tw_vect, phrase_vect):
    sim = cosine_similarity(Tw_vect, phrase_vect)
    similaridad_mayor_umbral = False
    if sim>0.35:
        similaridad_mayor_umbral = True
    return similaridad_mayor_umbral




model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

path = 'March_vectorize/'
listdirect = os.listdir(path)

#df0 = pd.DataFrame()
df_list = []
for df_Tw_vect in listdirect:
    df = pd.read_csv(df_Tw_vect)
    phr_vect = 0
    df['cluster'] = df.apply(lambda x: comparation(x['vectores'], phr_vect), axis = 1)
    df_cluster = df[df['cluster'] == True]
    df_list.append(df_cluster[['col1', 'col2', 'col3']])
    #df_nuevo = pd.concat([df0, df_cluster])

df_nuevo = pd.concat(df_list)
df_nuevo.to_csv('Tw_March_1_Cluster.csv')
