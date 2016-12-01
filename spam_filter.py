"""

Tillämpad Programmering, HT 2016
Fatima Guseinova
______________________________________

Programmet använder sig av enron3-mappen i min Mumin-mapp,
men det går enkelt att ändra filkällan till en annan mapp i main-delen.


"""


import os, random, nltk
from nltk import word_tokenize, WordNetLemmatizer, NaiveBayesClassifier, classify
from nltk.corpus import stopwords
from collections import Counter # så att jag kan använda mig av bag-of-words-modellen


icke_ord = stopwords.words ('english')

def fixa_listor (mapp): # inläsning av data
    """ läser in spam- respektive OK-mejl i var sin lista,
    listorna skapas i main-delen"""
    
    lista_med_filer = os.listdir(mapp)
    lista = []
    for fil in lista_med_filer:
        a = open (mapp + fil, mode='r', encoding='latin-1') # här avhjälps encoding-problemen
        lista.append (a.read())
    a.close()
    return lista



def förprocessa (mening): # normalisering av data
    
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(ordet.lower()) for ordet in word_tokenize(mening)]



def definiera_dragen (text, setting): # definierar dragen för vad som utgör ett spammejl

    if setting == 'bow': # ser till ordfrekvensen
        return {ordet: räkna for ordet, räkna in Counter (förprocessa(text)).items() if not ordet in icke_ord}
    else:
        return {ordet: True for ordet in förprocessa(text) if not ordet in icke_ord}    



def öva (drag, uppdelning): # en uppdelning av data i övnings- och testgrupper och träningsprocessen görs
    
    träningsmängd = int (len(drag) * uppdelning)

    
    # övningen på tränings- och dataseten påbörjas
    träningsdata, testdata = drag[:träningsmängd], drag[träningsmängd:]

    print (str(len(träningsdata)) + ' emails are being used in the training data, and \n' +
           str(len(testdata)) + ' emails are being used in the test data.\n')

    klassifieraren = NaiveBayesClassifier.train(träningsdata) # övar upp klassifieraren

    return träningsdata, testdata, klassifieraren



def utvärdera (träningsdata, testdata, klassifieraren):
    # utvärderar klassifierarens prestation med tränings- och testdatan som grund

    print ('Training accuracy: ' + str (classify.accuracy(klassifieraren, träningsdata)) + '.\n' +
           'Test accuracy: ' + str (classify.accuracy(klassifieraren, testdata)) + '.\n\n')

    klassifieraren.show_most_informative_features (20) # rapport på de 20 mest 'avslöjande' orden
    
    print ("\n\n                        There you go, I'm done!\n")



if __name__ == "__main__":

    print ('Let me see...\n')

    spam = fixa_listor('enron3/spam/')
    ham = fixa_listor('enron3/ham/')
    
    alla_mail = [(email, 'OK') for email in ham]
    alla_mail += [(email, 'spam') for email in spam]
    random.shuffle (alla_mail)
    
    alla_drag = [(definiera_dragen (email, ' '), label) for (email, label) in alla_mail]

    print ('I have found ' + str (len (alla_drag)) + ' feature sets.\n')
    
    träningsdata, testdata, klassifieraren = öva(alla_drag, 0.8) # använder 80%-20%-uppdelning av tränings- och testdata
    
    utvärdera (träningsdata, testdata, klassifieraren)
