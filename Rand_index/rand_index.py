import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pk
import itertools

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

pathGit = "c:/Git_proyects/Tesis/Embbedings_Lkvec/"
path = "c:/Facultad/Tesis/"
Lkvec = pd.read_csv(path+'Lkvec_all_in.csv')
Lkvec_unique = Lkvec.drop_duplicates(subset=['phrase'])

c_nuevo = pk.load(open(pathGit+'clusters_threshold_75_22.pk', 'rb'))
phr = pk.load(open(path+'phrases_to_emb.pickle', 'rb'))


cluster_mapping = {phr[j]: str(i+1) for i, clus in enumerate(c_nuevo) for j in clus}
Lkvec_unique['id_cluster'] = Lkvec_unique['phrase'].map(cluster_mapping)

Lkvec_ids = np.array(Lkvec_unique['id'])
Clusters_ids = np.array(Lkvec_unique['id_cluster'])

print(calculate_rand_index(Lkvec_ids, Clusters_ids))