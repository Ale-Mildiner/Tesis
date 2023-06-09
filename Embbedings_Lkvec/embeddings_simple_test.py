#%%
import numpy as np
import pandas as pd
import pickle as pk
from sentence_transformers import SentenceTransformer
import requests
from sklearn.metrics.pairwise import cosine_similarity
#%%
frases = ['llamar a las cosas por su nombre', 'nombrar a los objetos por su denominación', 'el auto es muy espaciado', 'el carro es muy grande']
#model = SentenceTransformer('Recognai/bert-base-spanish-wwm-cased-xnli')

model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
#model = SentenceTransformer('all-MiniLM-L6-v2')

frases_emb = model.encode(frases, show_progress_bar=True, convert_to_tensor=True)
similarity = cosine_similarity(np.array(frases_emb))