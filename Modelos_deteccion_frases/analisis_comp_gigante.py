import numpy as np
import pandas as pd
import pickle as pk
from sentence_transformers import SentenceTransformer, util
import requests

model = SentenceTransformer('all-MiniLM-L6-v2')
phrases_cg  = pk.load('componente_gigante.pickle')
phrases_emb = model.encode(phrases_cg, convert_to_tensor=True)

clusters = util.community_detection(phrases_emb, threshold=0.6)

pk.dump(clusters,open('clusters_componentegigante_threshold_75', 'wb'))

TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
chat_id = "6045013691"
message = f"Termine de correr el codigo que clusteriza con 0.6 a los embbedings de las frases"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"                                                                                                                                                    
print(requests.get(url).json()) # this sends the message