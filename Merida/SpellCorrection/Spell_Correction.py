from textblob import TextBlob
from textblob import Word 
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown'])
from spellchecker import SpellChecker
from termcolor import colored


def SpellCheck2(data):
    spell = SpellChecker()
    Spell_Words = []
# Note that this does not necessarily deal with punctuation unless you provide
# a custom tokenizer
    words_split = nltk.word_tokenize(data) 
    # misspelled = spell.unknown(words_split)
    for word in words_split:
        spell.word_frequency.load_words(['molded','.', '(',')'])
        correction = spell.correction(word)
        # if correction != word:  
        #     correction = colored(correction)
        Spell_Words.append(correction)
    
    Corrected_Words = TreebankWordDetokenizer().detokenize(Spell_Words)
    return Corrected_Words