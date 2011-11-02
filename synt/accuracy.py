# -*- coding: utf-8 -*-
import nltk.classify.util
from synt.utils.db import get_samples, RedisManager
from synt.utils.text import normalize_text
from synt.utils.extractors import get_extractor
from synt.guesser import Guesser 

def test_accuracy(db_name='', test_samples=0, neutral_range=0, offset=0, redis_db=5):
    """
    Returns two accuracies and classifier:
    NLTK accuracy is the internal accuracy of the classifier 
    Manual Accuracy is the accuracy when compared to pre-flagged/known samples and label.
   
    Keyword Arguments:
    db_name (str) -- Samples database to use, by default this is the same as your trained database 
                     with an offset to ensure unseen data. Should be a string database name located in ~/.synt. 
    
    test_samples (int) -- Amount of samples to use, by default this will be 25% of the training set amount. 
    
    neutral_range (float) -- Will be used to drop "neutrals" to see how real-world accuracy will look. 
                             For example in the case where neutral range is 0.2 if the sentiment 
                             guessed is not greater than 0.2 or less than -0.2 it is not considered.
                             Leaving this set to 0 will not cause the special case drops and will by default
                             categorize text as either positive or negative. This may be undesired as the classifier
                             will treat 0.0001 as positive even though it is not a strong indication.
    
    offset (int) -- By default the offset is decided from the end of the the trained amount, i.e 
                    if you've trained on 1000 and you have 250 testing samples the samples retrieved
                    will be from 1000-1250, you can override this offset if you wish to use a different
                    subset. 
    
    redis_db (int) -- The redis database to use. 
    """
    
    m = RedisManager(db=redis_db)
    trained_classifier = m.r.get('trained_classifier') #retrieve the trained classifier
    
    if not trained_classifier:
        print("Accuracy needs a classifier, have you trained?")
        return

    classifier = m.pickle_load(trained_classifier)
    
    #we want to make sure we are testing on a new set of samples therefore
    #we use the trained_to as our offset and proceed to use the samples
    #thereafter, unless an offset is otherwise specified
    trained_to = int(m.r.get('trained_to'))

    if not offset:
        offset = trained_to 
 
    if test_samples <= 0: #if no testing samples provided use 25% of our training number
        test_samples = int(trained_to * .25)
   
    if not db_name:
        db_name = m.r.get('trained_db') #use the trained samples database
    
    test_samples = get_samples(db_name, test_samples, offset=offset)
   
    testfeats = []
    trained_ext = m.r.get('trained_extractor')
    
    feat_ex = get_extractor(trained_ext)()

    #normalization and extraction
    for text, label in test_samples:
        tokens = normalize_text(text) 
        bag_of_words = feat_ex.extract(tokens) 

        if bag_of_words:
            testfeats.append((bag_of_words, label))
    
    nltk_accuracy = nltk.classify.util.accuracy(classifier, gold=testfeats) * 100 # percentify

    total_guessed = 0
    total_correct = 0
    total_incorrect = 0
    
    g = Guesser(extractor_type=trained_ext)
   
    #compare the guessed sentiments with our samples database to determine manual accuracy
    for text, label in test_samples:
        guessed = g.guess(text)
        if abs(guessed) < neutral_range:
            continue
        
        if (guessed > 0) == label.startswith('pos'):
            total_correct += 1
        else:
            #print text, label, guessed
            total_incorrect += 1

        total_guessed += 1
   
    assert total_guessed, "There were no guesses, make sure you've trained on the same database you're testing."
   
    manual_accuracy =  total_correct * 100.0 / total_guessed

    #TODO: precision and recall

    return (nltk_accuracy, manual_accuracy, classifier)

if __name__ == "__main__":
    #example accuracy
    import time

    neutral_range = 0.2
    redis_db      = 4

    print("Testing accuracy with neutral range: {}.".format(neutral_range))
    start = time.time() 
    
    n_accur, m_accur, c = test_accuracy(neutral_range=neutral_range, redis_db=redis_db)

    c.show_most_informative_features(30)

    print("NLTK Accuracy: {}".format(n_accur))
    print("Manual Accuracy: {}".format(m_accur))

    print("Successfully tested in {} seconds.".format(time.time() - start))
