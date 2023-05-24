from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

path = 'c:/Facultad/Tesis/'
root_emb = pickle.load(open(path +'root_emb.pickle', 'rb'))
similarity = cosine_similarity(np.array(root_emb))
pickle.dump(similarity, open(path+'root_similarity.pickle', 'wb'))