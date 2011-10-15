# -*- coding: utf-8 -*-
"""Tools for extracting features and text processing."""

from nltk.corpus import stopwords
from synt.utils.db import RedisManager

class WordExtractor(object):
    
    def extract(self, words):
            
        if not words: return

        return dict([(word, True) for word in words])

class StopWordExtractor(WordExtractor):
    
    def __init__(self, stop_words=None):
        if stop_words:
            self.stop_words = stop_words
        else:
            self.stop_words = set(stopwords.words('english'))

    def extract(self, words):
        assert self.stop_words, "This extractor relies on a set of stopwords."
        
        if not words: return

        return dict([(word,True) for word in words if word not in self.stop_words])

class BestWordExtractor(WordExtractor):
    
    def __init__(self, best_words=None):
        if best_words:
            self.best_words = best_words
        else:
            self.best_words = RedisManager().get_best_features()

    def extract(self, words):
        assert self.best_words, "This extractor relies on best words."
        
        if not words: return
        
        return dict([(word, True) for word in words if word in self.best_words])
