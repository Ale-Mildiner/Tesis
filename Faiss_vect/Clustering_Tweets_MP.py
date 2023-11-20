import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk
import re 
from sentence_transformers import SentenceTransformer, util
import faiss
import multiprocessing as mp

encoder = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")


path = ''
phrases = pk.load(open(path+'citas_media_cluster.pk', 'rb'))

Tw_mes = pd.read_csv(path+'Tweets_October.csv')
Tw_mes  = Tw_mes.dropna(subset=['Tw_limpios'])
Tw_mes = Tw_mes.reset_index()

subdivition = [0,1000000, 2000000, 3000000, 5000000, 6000000, len(Tw_mes)]


def plot_tw(df, phrase, index):
    df['Fecha'] =  pd.to_datetime(df['Fecha'], format='%a %b %d %H:%M:%S +0000 %Y')
    res_index = df.set_index('Fecha')
    df_count_hora = res_index.resample('1H').count()
    df_suavizado = df_count_hora.rolling('1D', center = True).mean()
    df_suavizado_2 = df_count_hora.ewm( alpha = 0.1).mean()
    plt.figure()
    plt.title(phrase)
    plt.plot(df_count_hora.index, df_count_hora['Tweets'], label = 'crudo', marker = '.') #Test comparation
    plt.plot(df_suavizado.index, df_suavizado['Tweets'], label = 'rolling', marker = '.', linestyle = 'solid')
    plt.plot(df_suavizado_2.index, df_suavizado_2['Tweets'], label = 'filltro ewp', marker = '.', linestyle = 'solid', color = 'k')
    plt.xticks(rotation = 45)
    plt.legend()
    plt.grid(alpha = 0.7)
    plt.savefig(path_imagenes+'imagenes_{index}.png')
    plt.close()
    return 0


def Searching_Tweets(search_text):
    search_vector = encoder.encode(search_text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    df = pd.DataFrame(columns=['distances', 'Tweets', 'index', 'id', 'Fecha'])
    for i in range(len(subdivition) - 1):
        index = faiss.read_index(path+f'vect_faiss_{i}')
        
        tw_file = Tw_mes[subdivition[i]:subdivition[i+1]]
        tw_file = tw_file.reset_index()
        
        k = index.ntotal
        distances, ann = index.search(_vector, k=k)

        results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
        
        merge = pd.merge(results,tw_file, left_on='ann', right_index=True)
        res = merge[merge['distances']<0.93]
        df  = pd.concat([df, res[['distances', 'Tweets', 'index', 'id', 'Fecha']]])

        df = df.drop_duplicates(subset=['id'])
        del index
    
    index = np.where(np.array(phrases) == search_text)[0][0]
    df.to_csv(path+f'ext_folder/Tw_cluster_{index}.csv')
    plot_tw(df, search_text, index)
    return 0

if __name__ == '__main__':
    pool = mp.Pool(process = 16)
    #resultados.append(pool.map(Searching_Tweets, phrases[:1500]))
    pool.map(Searching_Tweets, phrases[:1500])
    pool.close()
    pool.join()