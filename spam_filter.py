import nltk # flera paket som används härifrån
import os # för att kunna läsa filerna i mappen iterativt
import random
from collections import Counter # så att jag kan använda mig av bag-of-words-modellen
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
#from nltk import classify


# använder mig av mappen 'enron5' som innehåller drygt 3000 ham-mejl och hälften så många spam-mejl


# LADDA DATAN -> FÖRPROCESSA -> TA FRAM FEATURES -> TRÄNA KLASSIFIERAREN -> UTVÄRDERA

# förprocessning: tokenisering och lemmatisering samt gemene bokstäver för alla ord

# FEATURES:
# 1) ta bort icke-ord/stopp-ord,
# 2) för varje icke-ord - räkna hur frekvent det är i texten
# (härmed kan klassifieraren regga att vissa ord kan förekomma i båda emejltyper men olika ofta),
# 3) extrahera feautres från emejls och para ihop de med emejltaggen 'ham' eller 'spam'

icke_ord_lista = stopwords.words ('english')

def fixa_listor (mapp):
    fillista = os.listdir(mapp)
    lista = []
    for fil in fillista:
        a = open (mapp + fil, 'r')
        lista.append (a.read())
    a.close()
    return lista

# wordnetlemmatizer-modulen returnerar ett oförändrat inputord om det inte är funnet i WordNet
# tokenize-mmodulen returnerar lista av strängar
def förprocessa (mening):
    wnl = WordNetLemmatizer ()
    return [wnl.lemmatize(word.lower()) for word in word_tokenize (unicode (mening, errors = ignore))]



# parametern 'setting' tillåter mig styra vilken approach/modell jag vill nyttja
# defaultmodellen här är ordfrekvens
# provar att använda mig av "bag-of-words"-modellen som tillåter klassifieraren att
# regga att en del ord förekommer i både spam och ham, där de är mer frekventa i den ena än den andra gruppen

def feat_definition (text, setting):
    if setting == 'bow':
        return {word: count for word, count in Counter (förprocessa(text)).items() if word not in icke_ord_lista}
    else:
        return {word: True for word in förprocessa(text) if word not in icke_ord_lista}



#####################
#main-delen#
#####################

# preppar datan för vidare arbete
spam_mejl = fixa_listor('enron5/ham/')
ham_mejl = fixa_listor('enron5/ham/')


# preppar featuresen

# tränar klassifieraren

# utvärdering
