import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from numpy.linalg import norm
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

def vectorize(Tweets):
    return model.encode(Tweets, convert_to_tensor=True)

path = ''
Tweets_march = pd.read_csv(path+'Tweets_march.csv')
Tweets_march['vectores'] = Tweets_march['Tweets'].apply(vectorize)
Tweets_march.to_csv(path+'Tweets_march_vect.csv')
