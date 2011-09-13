"""Tools for extracting features and text processing."""

from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from synt.utils.redis_manager import RedisManager
import itertools

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
    
    best_words = RedisManager().get_best_words()
    if not (words and best_words): return
    return dict([(word, True) for word in words if word in best_words])

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    """Word features with bigrams and best words."""
    
    if not words: return
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d
