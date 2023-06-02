#%%
import numpy as np
import pandas as pd
import pickle as pk
import torch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
import requests

path = 'd:/Facultad/Tesis/'
all_data = pd.read_csv(path+'all_data.csv')
phrases = list(all_data['phrases'])

phr_emb = pk.load(open(path+'phr_embbedings/phrases_lkvec_emb.pickle', 'rb'))
phr = pk.load(open(path+'phr_embbedings/phrases_to_emb.pickle', 'rb'))
clusters = util.community_detection(phr_emb,threshold=0.75)

pk.dump(clusters,open('clusters_threshold_75', 'wb'))

TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
chat_id = "6045013691"
message = f"Termine de correr el codigo que clusteriza con 0.75 a los embbedings de las frases"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"                                                                                                                                                    
print(requests.get(url).json()) # this sends the message

#%%
path = 'd:/Facultad/Tesis/'

clusters = pk.load(open('clusters_threshold_8.pickle', 'rb'))
phr = pk.load(open(path+'phr_embbedings/phrases_to_emb.pickle', 'rb'))

#%%
for c in clusters[515]:
    print(phr[c])

#%%
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", phrases[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", phrases[sentence_id])
