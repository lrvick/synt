# -*- coding: utf-8 -*-
from synt.utils.redis_manager import RedisManager
from synt.utils.extractors import best_word_feats
from synt.utils.text import sanitize_text

DEFAULT_CLASSIFIER = RedisManager().load_classifier()

def guess(text, classifier=DEFAULT_CLASSIFIER, feat_ex=best_word_feats):
    """
    Takes a blob of text and returns the sentiment score (-1.0 - 1.0).
    
    Keyword Arguments:
    classifier      -- the classifier to use  (Note: for now we only have a naivebayes classifier)
    feat_ex         -- the feature extractor to use i.e bigram_word_feats, stopword_feats, found in extractors
    """

    assert classifier, "Needs a classifier."
    
    tokens = sanitize_text(text)
    
    bag_of_words = feat_ex(tokens)
    
    if bag_of_words:
        
        prob = classifier.prob_classify(bag_of_words)
        
        #return a -1 .. 1 score
        score = prob.prob('positive') - prob.prob('negative')
       
        #if score doesn't fall within -1 and 1 return 0.0 
        #example: single words might return a heavily biased score like -9.8343
        if not (-1 <= score <= 1):
            return 0.0
        
        return score

