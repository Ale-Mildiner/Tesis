import numpy as np
import pandas as pd
import os
import json
import requests
import time
import multiprocessing as mp
import pickle as pk

#def agregar_RT(Tweeets_df):
#    retweets_data = []
#    end = '.txt'
#    path_sophy = '../../sophy/NYData2022/August_Data/'
#    list_dir = os.listdir(path_sophy)
#    for filename in (list_dir):
#        if filename.endswith(end) and filename != 'keywords.txt':
#            with open(path_sophy+filename, 'r', encoding='utf8') as fp:
#                for i, line in enumerate(fp):
#                    json_data = json.loads(line)
#                    tweet = json_data['text']
#                    if tweet.startswith('RT'):
#                        try:
#                            id_cluster = list(Tweeets_df['id'])
#                            original_id = json_data['retweeted_status']['id']
#                            if original_id in id_cluster:
#                                print('entre para agregar RT')
#                                filt = Tweeets_df['id'] == original_id
#                                Tweet = Tweeets_df[filt]['Tweets'].tolist()[0]
#                                retweets_data.append({
#                                    "id": original_id,
#                                    "Tweets": Tweet,
#                                    "distances": 0,
#                                    "Fecha": json_data['created_at'],
#                                    "index": 0,
#                                    "RT": True
#                                })               
#                        except KeyError:
#                            pass
#    return retweets_data
#
def agregar_RT(Tweeets_df):
    retweets_data = []
    end = '.txt'
    path_sophy = '../../sophy/NYData2022/August_Data/'
    list_dir = os.listdir(path_sophy)


    id_cluster = set(Tweeets_df['id'])

    for filename in (list_dir):
        if filename.endswith(end) and filename != 'keywords.txt':
            with open(path_sophy+filename, 'r', encoding='utf8') as fp:
                for line in fp:
                    json_data = json.loads(line)
                    retweeted_status = json_data.get('retweeted_status', None)
                    try:
                        if retweeted_status['id'] in id_cluster:
                            if json_data.get('text', '').startswith('RT'):
                                print('entre')
                                original_id = retweeted_status['id']
                                filt = Tweeets_df['id'] == original_id
                                Tweet = Tweeets_df.loc[filt, 'Tweets'].tolist()[0]
                                retweets_data.append({
                                    "id": original_id,
                                    "Tweets": Tweet,
                                    "distances": 0,
                                    "Fecha": json_data['created_at'],
                                    "index": 0,
                                    "RT": True
                                })
                    except:
                        pass

    return retweets_data
#def agregar_RT(Tweeets_df):

             #   for line in fp:
             #       try:
             #           json_data = json.loads(line)
             #       except:
             #           print('no puedo leer linea')
             #       tweet = json_data['text']
             #       if tweet.startswith('RT'):
             #           try:
             #               id_cluster = list(Tweeets_df['id'])
             #               original_id = json_data['retweeted_status']['id']
             #               if original_id in id_cluster:
             #                   print('entre para agregar RT')
             #                   filt = Tweeets_df['id'] == original_id
             #                   Tweet = Tweeets_df[filt]['Tweets'].tolist()[0]
             #                   retweets_data.append({
             #                       "id": original_id,
             #                       "Tweets": Tweet,
             #                       "distances": 0,
             #                       "Fecha": json_data['created_at'],
             #                       "index": 0,
             #                       "RT": True
             #                   })               
             #           except KeyError:
             #               pass
    #return retweets_data
#def agregar_RT(Tweeets_df):
#    retweets_data = []
#    end = '.txt'
#    path_sophy = '../../sophy/NYData2022/August_Data/'
#    list_dir = os.listdir(path_sophy)
#
#    # Obtén la lista de ids una vez fuera del bucle
#    id_cluster = list(Tweeets_df['id'])
#
#    for filename in (list_dir):
#        if filename.endswith(end) and filename != 'keywords.txt':
#            with open(os.path.join(path_sophy, filename), 'r', encoding='utf8') as fp:
#                data = json.load(fp)
#                for json_data in data:
#                    tweet = json_data.get('text', '')
#                    if tweet.startswith('RT'):
#                        try:
#                            original_id = json_data['retweeted_status']['id']
#                            if original_id in id_cluster:
#                                print('entreeeeee')
#                                filt = Tweeets_df['id'] == original_id
#                                Tweet = Tweeets_df[filt]['Tweets'].tolist()[0]
#                                retweets_data.append({
#                                    "id": original_id,
#                                    "Tweets": Tweet,
#                                    "distances": 0,
#                                    "Fecha": json_data['created_at'],
#                                    "index": 0,
#                                    "RT": True
#                                })               
#                        except KeyError:
#                            pass
#    return retweets_data

path_sophy = '../../sophy/NYData2022/August_Data/'
list_dir = os.listdir(path_sophy)

def process_data(filename):
    #Tweeets_df = pd.read_csv(path+'Tw_'+str(i)+'_Cluster.csv')
    retweets_data = []

    Tweeets_df = pd.read_csv('Tw_cluster_August/0_1500/'+filename)
    Tweeets_df['RT'] = False
    Tweeets_df = Tweeets_df[['id', 'Tweets', 'distances', 'Fecha', 'index', 'RT']]

    df = Tweeets_df.copy()
    df['Fecha'] =  pd.to_datetime(df['Fecha'], format='%a %b %d %H:%M:%S +0000 %Y')
    res_index = df.set_index('Fecha') 
    df_count_hora = res_index.resample('1H').count()
    df_suavizado = df_count_hora.rolling('1D', center = True).mean()



    fecha_max = pd.to_datetime(df_count_hora[df_count_hora['Tweets'] == df_count_hora['Tweets'].max()].index.values[0])  
    fecha_inicio = pd.to_datetime('2019-08-09')
    fecha_fin = pd.to_datetime('2019-08-13') 
    if fecha_max > fecha_fin or fecha_max< fecha_inicio:



    #num_process = mp.cpu_count()
        retweets_data = agregar_RT(Tweeets_df)

    # with mp.Pool(num_process) as pool:
    #     results = pool.map(agregar_RT, list_dir)
    #     for res in results:
    #         retweets_data.append(res)

        df_RT = pd.DataFrame(retweets_data)
        Tweets_cluster_completo = pd.concat([Tweeets_df, df_RT], ignore_index=True)
        Tweets_cluster_completo.to_csv('Tw_mas_RT_all/RT_'+str(filename))
        TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
        chat_id = "6045013691"
        message = f"Termine de correr alguna iteración con nombre {filename}"
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url).json()
    else:
        print(f'ids on maximos {filename}')
        TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
        chat_id = "6045013691"
        message = f"cayo {filename}"
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url).json()



t0 = time.time()


if __name__ == '__main__':
    #path = 'Tw_citas_selected_2'
    list_csv = []
    #ids = pk.load(open('ids_august_2.pk', 'rb'))
    ids = np.arange(0,1500,1)
    for idd in ids:
        idd = str(idd)
        list_csv.append(f'Tw_cluster_{idd}.csv')
    #listdirect = os.listdir(path)
    #num_tareas = 
    pool = mp.Pool(processes=18)
    pool.map(process_data, list_csv)

    pool.close()
    pool.join()


    tf = time.time()


    TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
    chat_id = "6045013691"
    message = f" Termine de correr el codigo que me genera un df con los tweets clusterizados para un solo cluster 17,\n tardo {tf-t0} en correr"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json()) 
