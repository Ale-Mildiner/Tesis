import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import itertools
import requests
import time

t0 = time.time()
def calculate_rand_index(partitionA, partitionB):
    n = len(partitionA)
    a = 0  # Pares concordantes en la misma partición en ambas agrupaciones
    b = 0  # Pares concordantes en particiones diferentes en ambas agrupaciones

    for i in range(n-1):
        for j in range(i+1, n):
            if (partitionA[i] == partitionA[j]) and (partitionB[i] == partitionB[j]):
                a += 1
            elif (partitionA[i] != partitionA[j]) and (partitionB[i] != partitionB[j]):
                b += 1

    combinations = list(itertools.combinations(range(n), 2))
    num_combinations = len(combinations)

    rand_index = (a + b) / num_combinations
    return rand_index

pathGit = "d:/Git_proyects/Tesis/Embbedings_Lkvec/"
path = "d:/Facultad/Tesis/"
Lkvec = pd.read_csv(path+'Lkvec_all_in.csv')
Lkvec_unique = Lkvec.drop_duplicates(subset=['phrase'])

c_mnet = pk.load(open(pathGit+'clusters_threshold_75_22.pk', 'rb'))
c_mini = pk.load(open(pathGit+'clusters_threshold_75_02.pk', 'rb'))
phr = pk.load(open(path+'phr_embbedings/phrases_to_emb.pickle', 'rb'))


cluster_mapping_mnet = {phr[j]: str(i+1) for i, clus in enumerate(c_mnet) for j in clus}
Lkvec_unique['id_cluster_mnet'] = Lkvec_unique['phrase'].map(cluster_mapping_mnet)

cluster_mapping_mini = {phr[j]: str(i+1) for i, clus in enumerate(c_mini) for j in clus}
Lkvec_unique['id_cluster_mini'] = Lkvec_unique['phrase'].map(cluster_mapping_mini)

Lkvec_ids = np.array(Lkvec_unique['id'])
Clusters_ids_mnet = np.array(Lkvec_unique['id_cluster_mnet'])
Clusters_ids_mini = np.array(Lkvec_unique['id_cluster_mini'])

rand_index_mini = calculate_rand_index(Lkvec_ids, Clusters_ids_mini)
rand_index_mnet = calculate_rand_index(Lkvec_ids, Clusters_ids_mnet)

print('mnet', rand_index_mnet, '\n')
print('mini ',rand_index_mini, '\n')

tf = time.time()
print('tiempo', tf-t0)
TOKEN = "6287446315:AAFAnvbB6vUSzttp-smI5E00jDP7hNI7kCo" 
chat_id = ""
message = f"Termine con los siguientes valores \n mini = {rand_index_mini} \n mpnet = {rand_index_mnet} \n tiempo = {tf-t0}"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
print(requests.get(url).json()) # this sends the message     