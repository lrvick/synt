"""Tools for extracting features and text processing."""

import re
import string
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import WhitespaceTokenizer
from redis_manager import RedisManager
from BeautifulSoup import BeautifulSoup, BeautifulStoneSoup

from synt import settings

bestwords = RedisManager().best_words() 

def word_feats(words):
    """Basic word features, simple bag of words model"""

    return dict([(word, True) for word in words])

def stopword_word_feats(words):
    """Word features with stopwords"""

    stopset = set(stopwords.words('english'))
    return dict([(word,True) for word in words if word not in stopset])

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200, withstopwords=True):
    """Word features with bigrams"""
    if not words: return
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def best_word_feats(words):
    """Word feats with best words."""
    if not (words and bestwords): return
    return dict([(word, True) for word in words if word in bestwords])

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    """Word features with bigrams and best words."""
    if not words: return
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d

def sanitize_text(text):
    """
    Formats text to strip unneccesary:words, punctuation and whitespace. Returns a tokenized set.
    """
    
    if not text: return 
   
    text = text.lower()

    for e in settings.EMOTICONS:
        text = text.replace(e, '') #remove emoticons

    format_pats = (
        #match, replace with
        ("http.*", ''), #strip links
        ("@[A-Za-z0-9_]+", ''), #twitter specific ""
        ("#[A-Za-z0-9_]+", ''), # ""
        ("(\w)\\1{2,}", "\\1\\1"), #remove occurences of more than two consecutive repeating characters
    )
    
    for pat in format_pats:
        text = re.sub(pat[0], pat[1], text)
    
    try:
        text = str(''.join(BeautifulSoup(text).findAll(text=True))) #strip html and force str
    except Exception, e:
        print 'Exception occured:', e
        return


    text = text.translate(string.maketrans('', ''), string.punctuation).strip() #strip punctuation

    if text:
        words = [w for w in WhitespaceTokenizer().tokenize(text) if len(w) > 1]
    
        return words


