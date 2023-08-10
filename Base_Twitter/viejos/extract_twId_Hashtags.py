import json

''' # EN CASO DE TENER EL ARCHIVO EN OTRA CARPETA 
import os
path = os.getcwd()
print(path)
prev_path = path.replace('Codes','')
filename =prev_path + '/March_Data/201903-Macri.txt'
'''

filename = 'Muestra_201903-Macri.txt'

def take_url(json_data):
    try:
        url = json_data['urls'][0]['expanded_url']
    except:
        url = ''
    return url

file2save = open('hashtags_March_Macri.txt','w')
file2save.write('tw_id,hashtags\n')


with open(filename,"r") as fp:
    for line in fp:
        # Para cada linea lee el json y extrae la fecha
        json_data = json.loads(line)
        url = take_url(json_data)
        hashtags = json_data['hashtags']
        if len(hashtags)>0:
            #file2save.write(line) # En caso de guardad todo el json en el txt
            lista_hash = []
            tweet_id = json_data['id']
            line_hash = ''
            for l in hashtags:
                #lista_hash.append(l['text'])
                line_hash = line_hash + ' ' + l['text']
            file2save.write(str(tweet_id)+','+line_hash+'\n')
file2save.close()


