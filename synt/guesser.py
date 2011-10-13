# -*- coding: utf-8 -*-
from synt.utils.db import RedisManager
from synt.utils.extractors import WordExtractor, BestWordExtractor
from synt.utils.text import sanitize_text


class Guesser(object):
    
    def __init__(self, classifier='naivebayes', extractor=WordExtractor):
        
        self.classifier = RedisManager().load_classifier(classifier) 
        self.extractor = extractor()
        self.sanitizer = sanitize_text
    
    def guess(self, text):
        """
        Takes text and returns the sentiment score between -1 and 1.
        """
        
        if not self.classifier:
            print("guess needs a classifier.")
            return

        tokens = self.sanitizer(text)
      
        bag_of_words = self.extractor.extract(tokens)
       
        score = 0.0
        
        if bag_of_words:
            
            prob = self.classifier.prob_classify(bag_of_words)
            
            #return a -1 .. 1 score
            score = prob.prob('positive') - prob.prob('negative')
           
            #if score doesn't fall within -1 and 1 return 0.0 
            if not (-1 <= score <= 1):
                pass #score 0.0

        return score

guess = Guesser()

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
