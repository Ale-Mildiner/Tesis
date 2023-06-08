import os
import pandas as pd

path = ''

files =os.listdir(path+'Data_disgregada')
files.pop(0)
files
all_in_one = pd.read_csv(path+'Data_disgregada/'+files[0])
for i in range(len(files)-1):
    new = pd.read_csv(path+'Data_disgregada/'+ files[i+1])
    all_in_one = pd.concat([all_in_one, new], axis=0)
all_in_one.to_csv('Lkvec_all_in.csv')
