import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from numpy.linalg import norm
import requests

def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

def comparation(Tw_vect, phrase_vect):
    sim = cosine_similarity(Tw_vect, phrase_vect)
    similaridad_mayor_umbral = False
    if sim>0.45:
        similaridad_mayor_umbral = True
    return similaridad_mayor_umbral


def convert_to_array(columna):
    data_str = columna.replace(']','').replace('[','')
    data_list = data_str.split()
    data_arr = np.array(data_list, dtype = float)
    return data_arr



model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

path = 'Tweets_vect/'
listdirect = os.listdir(path)
phrase = 'Él era parte de un sistema que se vio extorsionado por el kirchnerismo en que para trabajar había que pagar'
phr_vect = model.encode(phrase)

#df0 = pd.DataFrame()
df_list = []
for df_Tw_vect in listdirect:
    df = pd.read_csv(path+df_Tw_vect, low_memory = False)
    df['vectores'] = df['vectores'].apply(convert_to_array)
    #print(cosine_similarity(np.array(df['vectores']), np.array(phr_vect)))
    df['cluster'] = df.apply(lambda x: comparation(x['vectores'], phr_vect), axis = 1)
    df_cluster = df[df['cluster'] == True]
    df_list.append(df_cluster[['Tweets', 'Fecha', 'urls', 'medios']])
    #df_nuevo = pd.concat([df0, df_cluster])
    del df

df_nuevo = pd.concat(df_list)
df_nuevo.to_csv('Tw_March_2_045_Cluster.csv', index = True)

TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
chat_id = "6045013691"
message = f" Termine de correr el codigo que me genera un df con los tweets clusterizados para un solo cluster"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
print(requests.get(url).json()) # this sends the message
