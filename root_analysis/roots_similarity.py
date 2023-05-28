from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

path = ''
root_emb = pickle.load(open(path +'root_emb.pickle', 'rb'))
similarity = cosine_similarity(np.array(root_emb))
np.fill_diagonal(similarity, 0)
filas, columnas = np.where(similarity >0.5)

pickle.dump(filas, open(path+'filas_index_05.pickle', 'wb'))
pickle.dump(columnas, open(path+'columnas_index_05.pickle', 'wb'))

#pickle.dump(similarity, open(path+'root_similarity.pickle', 'wb'))