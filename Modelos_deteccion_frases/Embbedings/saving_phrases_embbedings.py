import numpy as np
import pandas as pd
import pickle as pk
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
path = 'Tesis/Modelo_deteccion_frases/'
lkvec = pd.read_csv('all_data.csv')
phrases = lkvec['phrases']
phrases_emb = model.encode(phrases)
pk.dump(phrases_emb, open('phrases_lkvec_emb.pickle', 'wb'))