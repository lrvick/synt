# -*- coding: utf-8 -*-
import nltk.classify.util
from synt.utils.db import get_samples
from synt.utils.redis_manager import RedisManager
from synt.utils.text import sanitize_text
from synt.utils.extractors import best_word_feats
from synt.guesser import guess
from synt.logger import create_logger

logger = create_logger(__file__)

def test(test_samples=50000, feat_ex=best_word_feats):
    """
    This first returns the accuracy of the classifier then proceeds
    to test across known sentiments and produces a 'manual accuracy score'.
    
    Keyword Arguments:
    test_samples    -- the amount of samples to test against
    feat_ex         -- the feature extractor to use (utils/extractors)
    
    """

    man = RedisManager()
    classifier = man.load_classifier()
    
    if not classifier:
        logger.error("test needs a classifier")
        return

    results = []
    nltk_testing_dicts = []
    accurate_samples = 0
    
    logger.info("Preparing %s testing Samples" % test_samples)
    
    offset = man.r.get('training_sample_count')
    if not offset: offset = 0

    samples = get_samples(test_samples, offset=offset) #ensure we are using new testing samples
    
    for sample in samples:
        
        text, sentiment = sample[0], sample[1] #(text, sentiment)
        tokens = sanitize_text(text)
        
        if tokens:
            feats = feat_ex(tokens, best_words=man.get_best_words())
            
            nltk_testing_dicts.append((feats, sentiment))

    nltk_accuracy = nltk.classify.util.accuracy(classifier, nltk_testing_dicts)  * 100 # percentify
    
    for sample in samples:
        text, sentiment = sample[0], sample[1] #(text, sentiment)
        guessed = guess(text)
       
        if sentiment.startswith('pos') and guessed > 0:
            accurate = True
        elif sentiment.startswith('neg') and guessed < 0:
            accurate = True
        else:
            accurate = False
            
        
        results.append((accurate, sentiment, guessed, text))
    
    for result in results:
        
        logger.info("Test: %s | Accuracy: %s | Known Sentiment: %s | Guessed Sentiment: %s " %  
                (result[3], result[0], result[1], result[2]))
        
        if result[0] == True:
            accurate_samples += 1
       
        total_accuracy = (accurate_samples * 100.00 / len(samples)) 
    
    classifier.show_most_informative_features(30)
    logger.info("Manual classifier accuracy result: %s%%" % total_accuracy)
    logger.info("NLTK classifier accuracy result: %.2f%%" % nltk_accuracy)


if __name__ == "__main__":
    #example test on 100 samples
    test(100)
