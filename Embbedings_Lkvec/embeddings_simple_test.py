#%%
import numpy as np
import pandas as pd
import pickle as pk
from sentence_transformers import SentenceTransformer
import requests
from sklearn.metrics.pairwise import cosine_similarity
#%%
frases = ['El coche es grande', 'El automóvil tiene un tamaño considerable', 'me gusta mucho este libro', 'siento una gran atracción por esta obra literaria']
model = SentenceTransformer('all-MiniLM-L6-v2')
frases_emb = model.encode(frases, show_progress_bar=True, convert_to_tensor=True)
similarity = cosine_similarity(np.array(frases_emb))