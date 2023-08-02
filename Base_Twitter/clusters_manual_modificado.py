import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from numpy.linalg import norm
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

def recorrer_texto(df, n, m, threshold, root_emb):
    '''
    df: data_frame con notas/Tweets
    n: longitud de la frase con la que quiero comparar
    m: ventana temporal con la que quiero recorrer
    threshold: umbral de similaridad coseno a comparar
    fecha_inicio: desde que fecha
    fecha_fin: hasta que fecha
    return cantidad de veces que encuentra similitud mayor al threshold en todas las notas de esos días
    '''
    def extract_phrases(Tweets):
        palabras = Tweets.split()
        cluster = False
        #for inicio in range(0, len(palabras) - n + 1, m):
        inicio = 0
        fin = inicio + n
        while fin<len(palabras) and cluster == False:
            frases = " ".join(palabras[inicio : fin])
            frases_emb = model.encode(frases)
            sim = cosine_similarity(root_emb, frases_emb)
            if sim >= threshold:
                cluster = True
                inicio+=n-m
            
            inicio+=m
        return cluster

    cluster = df["Tweets_sin_url"].apply(extract_phrases)
    df['cluster']= cluster
    return df

root = "Hay gente que se dedica a vender droga porque se quedó sin laburo"
root_emb = model.encode(root)
path = 'd:/Git_Proyects/Tesis/Base_Twitter/'
df = pd.read_csv(path+'Tweets_kicillof_oct.csv')
df_nuevo = recorrer_texto(df, 13,2,0.75, root_emb)
df_nuevo