#%%
import numpy as np
import pandas as pd
import pickle as pk
import torch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


path = 'd:/Facultad/Tesis/'
all_data = pd.read_csv(path+'all_data.csv')
phrases = list(all_data['phrases'])

phr_emb = pk.load(open(path+'phrases_lkvec_emb.pickle', 'rb'))
ph_ee = torch.from_numpy(phr_emb)[:100000]
clusters = util.community_detection(ph_ee ,threshold=0.6)


for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", phrases[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", phrases[sentence_id])
