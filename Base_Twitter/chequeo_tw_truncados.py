import json
import os


def chequeo_trunc(filename):

    hay_trunc = False
    cant_trunc = 0

    with open(filename, 'r', encoding = "utf8") as fp:
        for i, line in enumerate(fp):
            json_data = json.loads(line)
            tweet = json_data['text']    
            if tweet[0:2] != 'RT':
                try:
                    if json_data.get('truncated') is True:
                        hay_trunc = True
                        cant_trunc += 1
                except:
                    pass
    return hay_trunc, cant_trunc


path = '../NYData2022/March_Data/'
list_dir = os.listdir(path)
end = '.txt'

truncado = []
cantidad_truncado = []

for i, filename in enumerate(list_dir):
    if filename.endswith(end) and filename!= "keywords.txt":
        trunc, cant_trunc = chequeo_trunc(filename)
        truncado.append(trunc)
        cantidad_truncado.append(cant_trunc)

print(truncado)
print(cantidad_truncado)