#%%
import numpy as np
import pandas as pd
import pickle as pk
import torch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


path = 'c:/Git-Proyects/Tesis/'
phr_emb = pk.load(open(path+'phrases_lkvec_emb.pickle', 'rb'))
ph_ee = torch.from_numpy(phr_emb)
clusters = util.community_detection(ph_ee,  min_community_size=6,threshold=0.81)
