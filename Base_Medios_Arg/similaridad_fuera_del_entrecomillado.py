#%%
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
#from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm


model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
text = "El gran desafío será cómo combinar agendas y el trabajo diario para continuar en la gestión de gobierno en los meses que faltan hasta diciembre y, al mismo tiempo, todo lo que conlleva estar en una campaña, que incluye viajes, actos y reuniones múltiples."

def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

def recorrer_texto(df, n, m, threshold, fecha_inicio, fecha_fin):
    '''
    df: data_frame con notas
    n: longitud de la frase con la que quiero comparar
    m: ventana temporal con la que quiero recorrer
    thrshold: umbral de similaridad coseno a comparar
    fecha_inicio: desde que fecha
    fecha_fin: hasta que fecha
    retunr cantidad de veces que encuentra similiaridad mayor al threshold en todas las notas de esos dias
    '''

    cont = 0
    fechas = pd.date_range(fecha_inicio,fecha_fin,freq='d')
    for fecha in fechas:
        for notas in df[(df['fecha'] >= fecha) & (df['fecha'] <= fecha)]['nota']:
            palabras = notas.split()
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
                
                # print(len(palabras_actuales), " ".join(palabras_actuales), '\n') 
                # print(inicio, fin)
                inicio += m
    
    return cont

root = 'Soy un europeísta. Soy alguien que cree en Europa. Porque de Europa, escribió alguna vez Octavio Paz, que los mexicanos salieron de los indios, los brasileros salieron de la selva, pero nosotros los argentinos llegamos de los barcos, y eran barcos que venían de allí, de Europa. Y así construimos nuestra sociedad'
root_emb = model.encode(root)
#%%
fecha_inicio = '2021-06-09'
fecha_fin = '2021-06-15'
print(pd.date_range(fecha_inicio,fecha_fin,freq='d'))

# path = 'd:/Facultad/Tesis/'
# base = pd.read_csv(path+'Corpus_medios_nac.csv')
# base['fecha'] = pd.to_datetime(base['fecha'])
# fecha_inicio = '2021-06-09'
# fecha_fin = '2021-06-09'
#threshold = 0.85
#contador = recorrer_texto(base, len(root), 5,threshold, fecha_inicio, fecha_fin)
#print(contador)
