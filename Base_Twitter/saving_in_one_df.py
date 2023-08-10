import json
import re
import pandas as pd
import numpy as np
import os


def take_url(json_data):
    try:
        url = json_data['urls'][0]['expanded_url']
    except:
        url = ''
    return url





medios_oficiales = ['Clarín', 'La Nación', 'El Litoral', 'Página|12', 'Bariloche Digital', 'Diario El Ciudadano', 'Radio Mitre', 'infobae', 'TN - Todo Noticias', 'Ámbito Financiero']
medios_oficiales_screnn = ['clarincom', 'lanacion', 'ellitoral', 'pagina12', 'barilochedigital', 'elciudadanoweb', 'radiomitre', 'infobae', 'todonoticias', 'Ambitocom']
medios_oficiales_screnn_2 = ['ellitoral', 'barilochedigital', 'pagina12', 'elciudadanodiario', 'elsigloweb', 'diarionorte', 'diariotextual', 'opisantacruz', 'eldiadelaplata', 'elciudadanoweb', 'chubutparatodos', 'tunoticia', 'diarionoticias', 'lagacetasalta', 'radiomitre', 'elzonda', 'jujuyaldia', 'santacruzalmomento', 'eldoce', 'tncorrientes', 'ultimahora', 'elpregon', 'misionesonline', 'informatesalta', 'losandes', 'laprensa', 'losprimerostv', 'diariouno', 'corrienteshoy', 'elesquiu', 'lamañanaformosa', 'app', 'infobae', 'lagaceta', 'lanacion', 'tn', 'clarin', 'laredlarioja', 'infomerlo', 'm24digital', 'elliberal', 'diariamente', 'chacodiapordia', 'informedigital', 'laprensafederal', 'elindependiente', 'vocescriticas', 'opinionciudadana', 'lamañanacordoba', 'elancasti', 'eltiemposanjuan', 'primeraedicion', 'telam', 'ambito', 'elcomodorense', 'surenio', 'lavoz']

def salvando_tweets(filename, path):
    dias = []
    frases = []
    url_reales = []
    medios = []
    with open(path+filename, 'r', encoding = "utf8") as fp:
        for i, line in enumerate(fp):
            # Para cada linea lee el json y extrae la fecha
            json_data = json.loads(line)
            url_real = take_url(json_data)
            tweet = json_data['text']

            if tweet[0:2] != 'RT':
                user = json_data['user']
                frases.append(tweet)
                dias.append(json_data['created_at'])


                try:
                    url_reales.append(url_real)
                except:
                    url_reales.append(0)
                try:
                    if user['name'] in medios_oficiales or user['screename'] in medios_oficiales_screnn or user['screename'] in medios_oficiales_screnn_2 :
                        medios.append(True)
                    else:
                        medios.append(False)
                except:
                    medios.append(False)

        df = pd.DataFrame({'Tweets': frases, 'Fecha': dias, 'urls': url_reales, 'medios': medios})


        return df

end = '.txt'
path  = '../NYData2022/March_Data/'
listdirect = os.listdir(path)
df0 = pd.DataFrame()
for i, filename in enumerate(listdirect):
    if filename.endswith(end) and filename!= "keywords.txt":
        print(filename)
        df = salvando_tweets(filename)
        df_nuevo = pd.concat([df0, df])
        df0 = df_nuevo
        numero_formateado = "{:02}".format(i)
        df_nuevo.to_csv('df_october/Tweets_october_'+str(numero_formateado)+'.csv')         