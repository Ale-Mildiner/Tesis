import numpy as np
import pandas as pd
import requests
import re
import os

path = ''
medios_nacionales = pd.read_csv(path+'Corpus_medios_nac.csv')
Tweets_mayo = pd.read_csv(path+'Tweets_march.csv')


links = list(medios_nacionales['link'])

def links_chek(link_df):
    is_in = False
    if link_df in links:
        is_in= True
    return is_in

Tweets_mayo['link_nacionales'] = Tweets_mayo['urls'].apply(links_chek)

Tweets_mayo.to_csv('Tweets_march_compare_urls.csv')

tw = Tweets_mayo[Tweets_mayo['link_nacionales']==True]
tw.to_csv('Tweets_march_solo_medios.csv')

print('tweets compelt',len(Tweets_mayo))
print('tweets que estan compartidos', len(tw))
