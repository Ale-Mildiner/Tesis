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
from collections import OrderedDict

def extrac_phrases(path, num):
    ids = np.arange(10, num)
    phrases = []
    id_save = []
    for id in ids:
        try:
            data = pd.read_csv(path + 'Data_disgregada/Lkvec_id'+str(id)+'.csv')  
            phrases = phrases + list(data['root'])
            id_save.append(id)
        except:
            pass
    return phrases, id_save

path = 'd:/Facultad/Tesis/'
phrases, id_save = extrac_phrases(path, 3000000)   
duration = 1000
freq = 440
winsound.Beep(freq, duration)
#%%

pickle.dump(list(set(phrases)), open(path+'root_phrases_2.pickle', 'wb'))
pickle.dump(id_save, open(path+'root_ids.pickle', 'wb'))

#%%
p= pickle.load(open('root_phrases.pickle', 'rb'))
#%% 2guardado y analisis paraguardar bien los ids, el otro los guarda en otro orden

# from collections import OrderedDict
# save = list(OrderedDict.fromkeys(phrases))
# pickle.dump(save, open(path+'root_phrases_2.pickle', 'wb'))
# #%% 

# roots_2 = pk.load(open(path1+'root_phrases_2.pickle', 'rb'))
# roots_ids = pk.load(open(path1+'root_ids.pickle', 'rb'))

# #%%
# ind = [0]*(len(roots))
# for i in range(len(roots)):
#     index = roots_2.index(roots[i])
#     ind[i] = index
# #%%
# idss = [0]*len(roots)
# for i, index in enumerate(ind):
#     idss[i] = roots_ids[index]


# pk.dump(idss, open(path1+'root_ids_bueno.pickle', 'wb'))
