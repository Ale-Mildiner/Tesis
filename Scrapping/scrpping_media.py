import requests
from bs4 import BeautifulSoup
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

def replace(x):
    x = str(x)
    x = x.replace('”', '"')
    x = x.replace('“', '"')
    x = x.replace('”', '"')
    x = x.replace('\x93', '"')
    x = x.replace('\x94', '"')
    x = x.replace('‘','"')    
    x = x.replace('’','"')    
    x = x.replace('‘', '"')
    x = x.replace('‘', '"')
    x = x.replace('’', '"')

    
def extract_quotes(x):
    return re.findall('"([^"]*)"', x)

def scrapper(url_):
    url = url_

    response = requests.get(url)
    fr = []
    if response.status_code == 200:

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            if len(paragraph.get_text()) >100:
                par = paragraph.get_text()
                if  extract_quotes(par) != []:
                    for i in extract_quotes(par):
                        fr.append(i)
    else:
        print(f'No se pudo acceder a la página. Código de estado: {response.status_code}')
    return fr

def expand_url(url):
    try:
        #r = requests.get(url, auth = ('user', 'pass'))
        r = requests.get(url).url
        #r= r.url
    except:
        r = url
    return r

path = 'd:/Git_Proyects/Tesis/Base_Twitter/Tw_meses/'
Tweets_mes = pd.read_csv(path+'Tweets_Agosto.csv')

Tw_urls = Tweets_mes.drop_duplicates(subset=['urls'])
Tw_urls = Tw_urls.dropna(subset=['urls'])

list_media = pd.read_csv(path+'../../Scrapping/list_media.csv', encoding='latin-1', delimiter=';')


Tw_red  = Tw_urls[0:100]
Tw_red['urls_exp'] = Tw_red['urls'].apply(expand_url)
Tw_red['media'] = Tw_red['urls_exp'].apply(lambda url: any(url.startswith(encabezado) for encabezado in list(list_media['Portal'])))

print(len(Tw_red[Tw_red['media'] ==True]))
