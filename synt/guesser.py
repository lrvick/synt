# -*- coding: utf-8 -*-
from synt.utils.redis_manager import RedisManager
from synt.utils.extractors import WordExtractor, BestWordExtractor
from synt.utils.text import sanitize_text
from synt.logger import create_logger

logger = create_logger(__file__)

DEFAULT_CLASSIFIER = RedisManager().load_classifier()

def guess(text, classifier=DEFAULT_CLASSIFIER, extractor=WordExtractor()):
    """
    Takes a blob of text and returns the sentiment score (-1.0 - 1.0).
    
    Keyword Arguments:
    classifier      -- the classifier to use 
    extractor       -- the feature extractor to use utils.extractors
    """
    
    if not classifier:
        logger.error("guess needs a classifier")
        return

    tokens = sanitize_text(text)
  
    bag_of_words = extractor.extract(tokens)
   
    score = 0.0
    
    if bag_of_words:
        
        prob = classifier.prob_classify(bag_of_words)
        
        #return a -1 .. 1 score
        score = prob.prob('positive') - prob.prob('negative')
       
        #if score doesn't fall within -1 and 1 return 0.0 
        if not (-1 <= score <= 1):
            pass #score 0.0

    return score

if __name__ == '__main__':
    #example guess
    print("Enter something to calucluate the synt of it!")
    print("Just press enter to quit.")
    
    running = True
    while running:
        text = raw_input("synt> ")
        if not text:
            break    
        print('Guessed: {}'.format(guess(text)))
