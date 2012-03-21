# -*- coding: utf-8 -*-
from synt.utils.db import RedisManager
from synt.utils.extractors import get_extractor
from synt.utils.text import normalize_text

class Guesser(object):

    def __init__(self, classifier_type='naivebayes', extractor_type='stopwords'): 
        self.classifier_type = classifier_type
        self.manager = RedisManager()
        self.extractor = get_extractor(extractor_type)()
        self.normalizer = normalize_text

    def _get_classifier(self):
        """
        Gets the classifier when it is first required.
        """
        self._classifier = self.manager.pickle_load(self.classifier_type)

    def guess(self, text):
        """
        Returns the sentiment score between -1 and 1.

        Arguments:
        text (str) -- Text to classify.

        """

        try:
            classifier = self._classifier
        except AttributeError:
            self._get_classifier()
            classifier = self._classifier
            assert classifier, "guess needs a classifier"

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
