#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk
import networkx as nx
import re 



path = "d:/Facultad/Tesis/"
base = pd.read_csv(path+'Corpus_medios_nac.csv', nrows=10000)
#base2 = pd.read_csv(path+'Corpus_medios_nac.csv', nrows=20, skiprows=range(1,100000))
#%%
def replace(x):
    return x.replace('“', '"')

def extract_quotes(x):
    return re.findall('"([^"]*)"', x)
#%%
base['nota'] = base['nota'].apply(replace)
base['citas']  = base['nota'].apply(extract_quotes)
#%%
def words_length(x):
    return len(x.split(" "))

df_quoutes2 = pd.DataFrame({'Fecha': base['fecha'].iloc[1:], 'Hora': base['hora'].iloc[1:], 'Citas': base['citas'].iloc[1:], 'Link': base['link'].iloc[1:], 'Nota': base['nota'].iloc[1:]})
df_quoutes = df_quoutes2.explode('Citas') # Cada cita sea una fila distinta
df_quoutes = df_quoutes.dropna(subset=['Citas']) # Elimino la filas que no haya quotes

#df_quoutes['Citas'] = df_quoutes['Citas'].apply(replace)

df_quoutes = df_quoutes.reset_index()
df_quoutes = df_quoutes.drop(['index'], axis = 1)
df_quoutes['Cant_Palabras'] = df_quoutes['Citas'].apply(words_length) # genero columna que cuente la cantidad de plabaras
df_quoutes_pf = df_quoutes[df_quoutes['Cant_Palabras'] > 4]


#%%
df_quoutes = pd.DataFrame(columns=['Fecha', 'Hora', 'Citas'])
for i in range(base.shape[0]-1):
    df = pd.DataFrame({'Fecha': base['fecha'][i+1], 'Hora': base['hora'][i+1], 'Cita': base['citas'][i+1]})
    df_quoutes = pd.concat([df_quoutes, df], ignore_index=True)
df_quoutes
# df2 = pd.DataFrame({'Fecha': base['fecha'][3], 'Hora': base['hora'][3], 'Cita': base['citas'][3]})
# df3 = pd.DataFrame({'Fecha': base['fecha'][96], 'Hora': base['hora'][96], 'Cita': base['citas'][96]})