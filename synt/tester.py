# -*- coding: utf-8 -*-
import nltk.classify.util
from synt.utils.db import get_samples
from synt.utils.redis_manager import RedisManager
from synt.utils.text import sanitize_text
from synt.utils.extractors import best_word_feats
from synt.guesser import guess

def test(test_samples=200000, feat_ex=best_word_feats):
    """
    This first returns the accuracy of the classifier then proceeds
    to test across known sentiments and produces a 'manual accuracy score'.
    
    Keyword Arguments:
    test_samples    -- the amount of samples to test against
    feat_ext        -- the feature extractor to use (utils/extractors)
    
    """

    classifier = RedisManager().load_classifier()
    
    if not classifier:
        print("There is not classifier in Redis yet, have you trained?")
        return

    results = []
    nltk_testing_dicts = []
    accurate_samples = 0
    
    print("Preparing %s Testing Samples" % test_samples)
    samples = get_samples(test_samples)
    
    for sample in samples:
        
        text, sentiment = sample[0], sample[1] #(text, sentiment)
        tokens = sanitize_text(text)
        
        if tokens:
            feats = feat_ex(tokens)
            
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
        print ("Text: %s" % (result[3]))
        print ("Accuracy: %s | Known Sentiment: %s | Guessed Sentiment: %s " % (result[0], result[1], result[2]))
        print ("------------------------------------------------------------------------------------------------------------------------------------------")
        
        if result[0] == True:
            accurate_samples += 1
       

        total_accuracy = (accurate_samples * 100.00 / len(samples)) 
    
    classifier.show_most_informative_features(30)
    print("\n\rManual classifier accuracy result: %s%%" % total_accuracy)
    print("\n\rNLTK classifier accuracy result: %.2f%%" % nltk_accuracy)


if __name__ == "__main__":
    #example test on 100 samples
    test(100)
