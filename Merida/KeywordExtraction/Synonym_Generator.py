import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet



def Synonym_Generation(word):
    synonyms = []
    antonyms = []



    for syn in wordnet.synsets("right"):
	    for l in syn.lemmas():
		    synonyms.append(l.name())
		    if l.antonyms():
			    	antonyms.append(l.antonyms()[0].name())

    print(set(synonyms))
    print(set(antonyms))