from nltk import word_tokenize
import string
import pylev

def find_common_words(phrase1, phrase2): #habria que elimiinar los singnos de puntuacion.
    phrase1 = phrase1.translate(str.maketrans('', '', string.punctuation))
    phrase2 = phrase2.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the phrases into individual words
    words1 = word_tokenize(phrase1.lower())
    words2 = word_tokenize(phrase2.lower())
    
    # Determine the length of the shorter phrase
    min_len = min(len(words1), len(words2))
    
    # Initialize variables to keep track of the number of consecutive common words
    max_common = 0
    curr_common = 0
    
    # Loop through the words in the phrases and count consecutive common words
    for i in range(min_len):
        if words1[i] == words2[i]:
            curr_common += 1
            max_common = max(max_common, curr_common)
        else:
            curr_common = 0
    
    # Return the maximum number of consecutive common words
    return max_common


frase1 = "Hola, esto es una prueba, veremos, si funca"
frase2 = "hola, esto es una prueba. Veremos, no creo que funque"
frase1_token = frase1.split(" ")
frase2_token = frase2.split(" ")
dist = pylev.levenschtein(frase1_token, frase2_token)
print(find_common_words(frase1, frase2))
print('distancia de levenschtein', dist)

