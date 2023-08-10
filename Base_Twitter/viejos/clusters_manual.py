import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from numpy.linalg import norm
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

def recorrer_texto(df, n, m, threshold, root_emb):
    '''
    df: data_frame con notas/Tweets
    n: longitud de la frase con la que quiero comparar
    m: ventana temporal con la que quiero recorrer
    thrshold: umbral de similaridad coseno a comparar
    fecha_inicio: desde que fecha
    fecha_fin: hasta que fecha
    retunr cantidad de veces que encuentra similiaridad mayor al threshold en todas las notas de esos dias
    '''

    cluster = []
    for i in range(len(df)):
        cont = 0
        Tweets = df.loc[i]['Tweets_sin_url']
        palabras = Tweets.split()
        inicio = 0
        fin = inicio + n
        while fin < len(palabras):
            fin = inicio + n
            palabras_actuales = palabras[inicio:fin]
            frases = " ".join(palabras_actuales)
            frases_emb = model.encode(frases)
            sim = cosine_similarity(root_emb, frases_emb)
            if sim >= threshold:
                cont += 1
                inicio += n-m
            
            # print(len(palabras_actuales), " ".join(palabras_actuales), '\n') 
            # print(inicio, fin)
            inicio += m
        if cont>0:
            cluster.append(True)
        else:
            cluster.append(False)
    
    return cluster

root = "Hay gente que se dedica a vender droga porque se quedó sin laburo"
root_emb = model.encode(root)
df = pd.read_csv('Tweets_kicillof_oct')
cluster = recorrer_texto(df, 13, 2, 0.75, root_emb)
df['cluster'] = cluster
df.to_csv('Tweets_kici_nuevo.csv')