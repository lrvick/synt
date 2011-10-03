# -*- coding: utf-8 -*-
import nltk.classify.util
from synt.utils.db import get_samples
from synt.utils.redis_manager import RedisManager
from synt.utils.text import sanitize_text
from synt.utils.extractors import best_word_feats
from synt.guesser import guess
from synt.logger import create_logger

logger = create_logger(__file__)

def test(test_samples=50000, feat_ex=best_word_feats, neutral_zone=0):
    """
    This first returns the accuracy of the classifier then proceeds
    to test across known sentiments and produces a 'manual accuracy score'.
    
    Keyword Arguments:
    test_samples    -- the amount of samples to test against
    feat_ex         -- the feature extractor to use (utils/extractors)
    neutral_zone    -- will be used to drop "neutrals" to see how real-world accuracy will look.
                       In other words, in the case of neutral zone being 0.2 if the word
                       guessed is not greater than 0.2 or less than -0.2 it is considered inaccurate.
                       Leaving this set to 0 will always force the classifier
                       to provide a positive or negaitve return even if it is unmeaningful
                       i.e a score of 0.00001 is still positive but the classifier is 
                       more than likely uncertain about it.
    """

    man = RedisManager()
    classifier = man.load_classifier()
    
    if not classifier:
        logger.error("test needs a classifier")
        return

    results = []
    testfeats = []
    accurate_samples = 0
    
    logger.info("Preparing %s testing Samples" % test_samples)
    
    offset = int(man.r.get('training_sample_count'))
    if not offset: offset = 0

    samples = get_samples(test_samples, offset=offset) #ensure we are using new testing samples
   
    best_words = man.get_best_words()
    for text, label in samples:
        s_text = sanitize_text(text) 
        tokens = feat_ex(s_text, best_words=best_words)

        if tokens:
            testfeats.append((tokens, label))

    nltk_accuracy = nltk.classify.util.accuracy(classifier, testfeats)  * 100 # percentify
    
    for text, label in samples:
        guessed = guess(text, best_words=best_words)
       
        if label.startswith('pos') and guessed > neutral_zone: 
            accurate = True
        elif label.startswith('neg') and guessed < -neutral_zone:
            accurate = True
        else:
            accurate = False
            
        
        results.append((accurate, label, guessed, text))
    
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
    #example test on 1000 samples
    test(50000, neutral_zone=0.3)
