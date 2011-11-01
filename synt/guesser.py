# -*- coding: utf-8 -*-
from synt.utils.db import RedisManager
from synt.utils.extractors import get_extractor
from synt.utils.text import normalize_text

class Guesser(object):
    
    def __init__(self, classifier_type='naivebayes', extractor_type='stopwords', redis_db=5):
        self.classifier_type = classifier_type
        self.redis_db = redis_db
        self.extractor = get_extractor(extractor_type)()
        self.normalizer = normalize_text
   
    def _get_classifier(self):
        print("Retrieving classifier ...")
        self._classifier = RedisManager(db=self.redis_db).pickle_load(self.classifier_type) 

    def guess(self, text):
        """
        Takes text and returns the sentiment score between -1 and 1.
        """
        
        try:
            classifier = self._classifier
        except AttributeError:
            self._get_classifier()
            classifier = self._classifier
            if not classifier:
                print("guess needs a classifier, make sure you're using a supported classifier and you've trained.")
                return

        tokens = self.normalizer(text)
      
        bag_of_words = self.extractor.extract(tokens)
       
        score = 0.0
        
        if bag_of_words:
            
            prob = classifier.prob_classify(bag_of_words)
            
            #return a -1 .. 1 score
            score = prob.prob('positive') - prob.prob('negative')
           
            #if score doesn't fall within -1 and 1 return 0.0 
            if not (-1 <= score <= 1):
                pass 

        return score

guess = Guesser().guess

if __name__ == '__main__':
    #example usage of guess

    g = Guesser()
    
    print("Enter something to calculate the synt of it!")
    print("Just press enter to quit.")
    
    while True:
        text = raw_input("synt> ")
        if not text:
            break    
        print('Guessed: {}'.format(g.guess(text)))
