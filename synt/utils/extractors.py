# -*- coding: utf-8 -*-
"""Tools for extracting features and text processing."""

from nltk.corpus import stopwords

try:
    stopwords.words
except LookupError:
    import nltk
    print("Downloading needed nltk data...")
    nltk.download('all')

from synt.utils.db import RedisManager

def get_extractor(type):
    """
    Return the extractor for type.
    
    Arguments:
    type (str) -- A name/type of extractor.
    
    """

    extractors = {
        'words'     : WordExtractor,
        'stopwords' : StopWordExtractor,
        'bestwords' : BestWordExtractor,
    }

    if type not in extractors:
        raise KeyError("Extractor of type %s doesn't exist." % type)
    return extractors[type]

class WordExtractor(object):
     
    def extract(self, words, as_list=False):
        """
        Returns a base bag of words.
        
        Arguments:
        words (list) -- A list of words.

        Keyword Arguments:
        as_list (bool) -- By default we return a dict, unless you want to leave it as a list. 
        
        """
        
        if not words: return
        
        if as_list:
            return [word for word in words]

        return dict([(word, True) for word in words])

class StopWordExtractor(WordExtractor):
    
    def __init__(self, stop_words=None):
        if stop_words:
            self.stop_words = stop_words
        else:
            self.stop_words = set(stopwords.words('english'))

    def extract(self, words, as_list=False):
        """
        Returns a bag of words for words that are not in stop words.
        
        Arguments:
        words (list) -- A list of words.

        Keyword Arguments:
        as_list (bool) -- By default we return a dict, unless you want to leave it as a list. 
     
        """
        
        assert self.stop_words, "This extractor relies on a set of stop words."
        
        if not words: return
        
        if as_list:
            return [word for word in words if word not in self.stop_words]

        return dict([(word,True) for word in words if word not in self.stop_words])

class BestWordExtractor(WordExtractor):
    
    def __init__(self, best_words=None):
        if best_words:
            self.best_words = best_words
        else:
            self.best_words = RedisManager().get_best_features()

    def extract(self, words, as_list=False):
        """
        Returns a bag of words for words that are in best words.
        
        Arguments:
        words (list) -- A list of words.

        Keyword Arguments:
        as_list (bool) -- By default we return a dict, unless you want to leave it as a list. 
     
        """
        
        assert self.best_words, "This extractor relies on best words."
        
        if not words: return
        
        if as_list:
            return [word for word in words if word in self.best_words]
   
        return dict([(word, True) for word in words if word in self.best_words])
