"""Tools for extracting features and text processing."""

from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from synt.utils.redis_manager import RedisManager
import synt.settings as settings

man = RedisManager()
if 'word_scores' in man.r.keys():
    BEST_WORDS = man.get_best_words()

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
    if not (words and BEST_WORDS): return
    return dict([(word, True) for word in words if word in BEST_WORDS])

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    """Word features with bigrams and best words."""
    if not words: return
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d
