import numpy as np
import pandas as pd
import os
import json
import requests
import time
import multiprocessing as mp

def agregar_RT(Tweeets_df):
    retweets_data = []
    end = '.txt'
    path_sophy = '../../sophy/NYData2022/March_Data/'
    list_dir = os.listdir(path_sophy)
    for filename in (list_dir):
        if filename.endswith(end) and filename != 'keywords.txt':
            with open(path_sophy+filename, 'r', encoding='utf8') as fp:
                for i, line in enumerate(fp):
                    json_data = json.loads(line)
                    tweet = json_data['text']
                    if tweet.startswith('RT'):
                        try:
                            id_cluster = set(list(Tweeets_df['id']))
                            original_id = json_data['retweeted_status']['id']
                            if original_id in id_cluster:
                                filt = Tweeets_df['id'] == original_id
                                medios = Tweeets_df[filt]['medios'].tolist()[0]
                                retweets_data.append({
                                    "Tweets": original_id,
                                    "Fecha": json_data['created_at'],
                                    "urls": 0,
                                    "medios": medios,
                                    "id": original_id,
                                    "RT": True
                                })               
                        except KeyError:
                            pass
    return retweets_data


path = 'Tw_200_citas_55/'
path_sophy = '../../sophy/NYData2022/March_Data/'
list_dir = os.listdir(path_sophy)

def process_data(i):
    Tweeets_df = pd.read_csv(path+'Tw_'+str(i)+'_Cluster_muchas.csv')
    Tweeets_df['RT'] = False
    Tweeets_df = Tweeets_df[['Tweets', 'Fecha', 'urls', 'medios', 'id', 'RT']]

 #   id_cluster = set(list(Tweeets_df['id']))

    retweets_data = []

    #num_process = mp.cpu_count()
    retweets_data = agregar_RT(Tweeets_df)

    # with mp.Pool(num_process) as pool:
    #     results = pool.map(agregar_RT, list_dir)
    #     for res in results:
    #         retweets_data.append(res)

    df_RT = pd.DataFrame(retweets_data)
    Tweets_cluster_completo = pd.concat([Tweeets_df, df_RT], ignore_index=True)
    Tweets_cluster_completo.to_csv('Tw_mas_RT/'+'Tweets_cluster_con_RT_'+str(i)+'_55.csv')

    TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
    chat_id = "6045013691"
    message = f"Termine de correr la iteración {i}"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json()) 


t0 = time.time()


if __name__ == '__main__':
    num_tareas = 199
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(process_data, range(num_tareas))

    pool.close()
    pool.join()


tf = time.time()


TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
chat_id = "6045013691"
message = f" Termine de correr el codigo que me genera un df con los tweets clusterizados para un solo cluster 17,\n tardo {tf-t0} en correr"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
print(requests.get(url).json()) 
