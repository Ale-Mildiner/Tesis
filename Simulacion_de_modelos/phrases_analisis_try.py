from nltk import word_tokenize
import string
import pylev
import networkx as nx
import matplotlib.pyplot as plt


def count_consecutive_words(phrase1, phrase2):
    phrase1 = phrase1.translate(str.maketrans('', '', string.punctuation))
    phrase2 = phrase2.translate(str.maketrans('', '', string.punctuation))

    # Split the phrases into words
    words1 = phrase1.split()
    words2 = phrase2.split()
    words1 = word_tokenize(phrase1.lower())
    words2 = word_tokenize(phrase2.lower())


    # Initialize a counter and maximum value to zero
    count = 0
    max_count = 0

    # Loop through each word in the first phrase
    for i in range(len(words1)):
        # Check if the current word is in the second phrase
        if words1[i] in words2:
            # Find the index of the word in the second phrase
            j = words2.index(words1[i])
            # Loop through the remaining words in both phrases and count the consecutive matching words
            while i < len(words1) and j < len(words2) and words1[i] == words2[j]:
                count += 1
                i += 1
                j += 1
            # Update the maximum count value
            max_count = max(max_count, count)
            # Reset the counter to zero
            count = 0
    # Return the maximum count of consecutive words
    return max_count


def make_graph(variables, variables_label):
    G = nx.DiGraph()
    G.add_nodes_from(variables_label)

    for i_txt, i in enumerate(variables):
        for j_txt, j in enumerate(variables):
            if i != j:
                if count_consecutive_words(i,j) >= 10 and len(i.split(" ")) < len(j.split(" ")):
                    G.add_edge(variables_label[i_txt],variables_label[j_txt])

    return G





frase1 = "Hola, esto es una prueba, veremos, si funca"
frase2 = "hola, esto es una prueba. Veremos, no creo que funque"
print(count_consecutive_words(frase1, frase2))


# frase1_token = frase1.split(" ")
# frase2_token = frase2.split(" ")
# dist = pylev.levenschtein(frase1_token, frase2_token)
#print('distancia de levenschtein', dist)

pagina = "Le he presentado al Presidente Alberto Fernández mi renuncia indeclinable como jefe de Asesores de manera inmediata"
infobae = "A raíz de los rumores que circularon desde anoche y a los efectos de desactivar cualquier operación tendiente a intranquilizar los mercados le he presentado al Presidente mi renuncia indeclinable como Jefe de Asesores de manera inmediata"
lanacion = "tendientes a intranquilizar los mercados"
ambito = "desactivar cualquier operación tendiente a intranquilizar los mercados"
cronista = "A raíz de los rumores que circularon desde anoche y a los efectos de desactivar cualquier operación tendiente a intranquilizar los mercados le he presentado al Presidente Alberto Fernández mi renuncia indeclinable como Jefe de Asesores de manera inmediata"
clarin = "A raíz de los rumores que circularon desde anoche y a los efectos de desactivar cualquier operación tendiente a intranquilizar los mercados le he presentado al Presidente @alferdez mi renuncia indeclinable como Jefe de Asesores de manera inmediata"
destape = "A raíz de los rumores que circularon desde anoche y a los efectos de desactivar cualquier operación tendiente a intranquilizar los mercados le he presentado al Presidente @alferdez mi renuncia indeclinable como Jefe de Asesores de manera inmediata"

print('medios digitales', count_consecutive_words(clarin, pagina))


variables = [pagina, infobae, lanacion, ambito, cronista, clarin, destape]
variables_label = ('P12', 'infobae', 'nacion', 'ambito', 'cronista', 'clarin', 'destape')

grafo = make_graph(variables, variables_label)
plt.figure()
nx.draw(grafo, with_labels =True)
plt.show()

## pruebo buscando similaridaes pero con una libreria que busca similaridades en el texto en si
import spacy

nlp = spacy.load("es_core_news_md")  # load the medium-sized Spanish language model

simil  = nlp(infobae).similarity(nlp(lanacion))
print('similaridad en texto', simil)

for txt, i in enumerate(variables):
    for txtj, j in enumerate(variables):
        if i != j:
            simil = nlp(i).similarity(nlp(j))
            print(f"similaridad entre {variables_label[txt]} y {variables_label[txtj]} es {simil}")