#%%
#from nltk import word_tokenize
#import string
#import pylev
#import networkx as nx
import pandas as pd
import numpy as np
import time
import winsound
import pickle

def extrac_phrases(path, num):
    ids = np.arange(10, num)
    phrases = []
    for id in ids:
        try:
            data = pd.read_csv(path + 'Data_disgregada/Lkvec_id'+str(id)+'.csv')  
            phrases = phrases + list(data['root'])
        except:
            pass
    return phrases

path = 'd:/Facultad/Tesis/'
phrases = extrac_phrases(path, 3000000)   
duration = 1000
freq = 440
winsound.Beep(freq, duration)
#%%

pickle.dump(phrases, open(path+'root_phrases.pickle', 'wb'))
#%%
p= pickle.load(open('root_phrases.pickle', 'rb'))
#%%
print(p)
