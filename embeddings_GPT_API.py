#%%
import openai
import os
import requests
import numpy as np
import pandas as pd
import pickle as pk

key = ""
openai.api_key = os.getenv(key)
openai.api_key = key
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

path = 'd:/Facultad/Tesis/'
phr = pk.load(open(path+'phr_embbedings/phrases_to_emb.pickle', 'rb'))
# print(phr[1])
# print(get_embedding(phr[1]))
#%%

embed = []
for i in range(30):
   embed.append(get_embedding(phr[i]))