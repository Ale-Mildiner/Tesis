import numpy as np
import pandas as pd
import os
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()

path = ''
path_sophy = '../../sophy/NYData2022/March_Data/'
Tweeets_df = pd.read_csv(path+ 'Tweets_cluster_1.csv')
Tweeets_df['RT'] = False


tweets_cluster = set(list(Tweeets_df['Tweets']))

retweets_dict = {tweet: [] for tweet in tweets_cluster}

list_dir = os.listdir(path)
end = '.txt'

retweets_data = []
for i, filename in enumerate(list_dir):
    if filename.endswith(end) and filename != 'keywords.txt':

        with open(filename, 'r', encoding='utf8') as fp:
            for j, line in enumerate(fp):
                json_data = json.loads(line)
                tweet = json_data['text']
                if tweet.startswith('RT'):
                    try:
                        original_tweet = json_data['retweeted_status']['full_text']
                        if original_tweet in tweets_cluster:
                            retweets_dict[original_tweet].append(tweet)
                            filt = Tweeets_df['Tweets'] == original_tweet
                            medios = Tweeets_df[filt]['medios'].tolist()[0]
                            retweets_data.append({
                                "Tweets": original_tweet,
                                "Fecha": json_data['created_at'],
                                "urls": 0,
                                #"urls_reales":0, 
                                "medios": medios,
                                #"Tweets_sin_url": original_tweet,
                                #"cluster": True,
                                "RT": True
                            })               
                    except KeyError:
                        pass


df_RT = pd.DataFrame(retweets_data)
Tweets_cluster_completo = pd.concat([Tweeets_df, df_RT], ignore_index=True)
Tweets_cluster_completo.to_csv('Tw_mas_RT'+'Tweets_cluster_con_RT.csv', ignore_index = True)


tf = time.time()


TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo"
chat_id = "6045013691"
message = f" Termine de correr el codigo que me genera un df con los tweets clusterizados para un solo cluster 17,\n tardo {tf-t0} en correr"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
print(requests.get(url).json()) 



def process_file(filename):
    retweets_data_local = []
    if filename.endswith(end) and filename != 'keywords.txt':
        with open(filename, 'r', encoding='utf8') as fp:
            for line in fp:
                json_data = json.loads(line)
                tweet = json_data['text']
                if tweet.startswith('RT'):
                    try:
                        original_tweet = json_data['retweeted_status']['full_text']
                        if original_tweet in tweets_cluster:
                            retweets_data_local.append({
                                "Tweets": original_tweet,
                                "Fecha": json_data['created_at'],
                                "urls": 0,
                                "medios": Tweeets_df[Tweeets_df['Tweets'] == original_tweet]['medios'].values[0],
                                "RT": True
                            })               
                    except KeyError:
                        pass
    return retweets_data_local
