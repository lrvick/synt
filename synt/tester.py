# -*- coding: utf-8 -*-
import nltk.classify.util
from synt.utils.db import get_samples, RedisManager
from synt.utils.text import sanitize_text
from synt.utils.extractors import WordExtractor, BestWordExtractor
from synt.guesser import Guesser 

def test(test_samples=50000, classifier='naivebayes', extractor=BestWordExtractor, neutral_range=0):
    """
    Returns two accuracies:
        NLTK accuracy is the internal accuracy of the classifier 
        Manual Accuracy is the accuracy when compared to pre-flagged/known samples and sentiment.
    
    Keyword Arguments:
    test_samples    -- The amount of samples to test against.
    classifier      -- The classifier to use. NOTE: only supports naivebayes currently
    extractor       -- The feature extractor to use. (found in utils.extractors)
    neutral_range   -- Will be used to drop "neutrals" to see how real-world accuracy will look.
                       For example in the case where neutral range is 0.2 if the sentiment 
                       guessed is not greater than 0.2 or less than -0.2 it is considered inaccurate.
                       Leaving this set to 0 will not cause the special case drops and will by default
                       categorize text as either positive or negative. This may be undesired as the classifier
                       will treat 0.0001 as positive even though it is not a strong indication.
    """

    rm = RedisManager()
    classifier = rm.load_classifier(classifier)
    
    if not classifier:
        print("test needs a classifier")
        return

    #we want to make sure we are testing on a new set of samples therefore
    #we use the training_sample_count as our offset and proceed to use the samples
    #thereafter
    offset = int(rm.r.get('training_sample_count'))
    if not offset: offset = 0

    samples = get_samples(test_samples, offset=offset)
   
    testfeats = []
    feat_ex = extractor()

    for text, label in samples:
        tokens = sanitize_text(text) 
        bag_of_words = feat_ex.extract(tokens) 

        if bag_of_words:
            testfeats.append((bag_of_words, label))

    nltk_accuracy = nltk.classify.util.accuracy(classifier, gold=testfeats) * 100 # percentify

    total_guessed = 0
    total_correct = 0
    
    g = Guesser(extractor=extractor)
    
    for text, label in samples:
        guessed = g.guess(text)
        if abs(guessed) < neutral_range:
            continue
        
        if (guessed > 0) == label.startswith('pos'):
            total_correct += 1
        
        total_guessed += 1
   
    assert total_guessed, "There were no guesses, make sure you've trained on the same database you're testing."
    
    manual_accuracy =  total_correct * 100.0 / total_guessed

    return nltk_accuracy, manual_accuracy, classifier

if __name__ == "__main__":
    #example test on 5000samples
    test_samples = 75000 

    print("Testing on {} samples.".format(test_samples))
    n_accur, m_accur, c = test(test_samples, neutral_range=0.2)

    c.show_most_informative_features(30)

    print("NLTK Accuracy: {}".format(n_accur))
    print("Manual Accuracy: {}".format(m_accur))
