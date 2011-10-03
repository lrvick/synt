# -*- coding: utf-8 -*-
from utils.redis_manager import RedisManager
from synt.utils.extractors import best_word_feats
from synt.utils.db import get_samples
from synt.utils.text import sanitize_text
from synt.logger import create_logger
import time

logger = create_logger(__file__)

def train(feat_ex=best_word_feats, train_samples=400000, word_count_samples=200000, \
    word_count_range=150000,  bestwords_to_store=10000, processes=8, force_update=False, verbose=True):
    """
    Trains a Naive Bayes classifier with samples from database and stores the 
    resulting classifier in Redis.
  
    Args:
    #TODO: Currently this works and is tested only with best_word_feats but this should 
    #eventually work with various extractors.
    
    feat_ex             -- the feature extractor to use, found in utils/extractors.py. 

    Keyword arguments:
    train_samples      -- the amount of samples to train half this number will be negative the other positive 
    word_count_samples -- the amount of samples to build word_counts, this produces a word:count histogram in Redis 
    word_count_range   -- the amount of 'up-to' words to use for the FreqDist will pick out the most
                         'popular' words up to this amount. i.e top 150000 tokens 
    bestwords_to_store -- the amount of of words we will use in our 'best_words' list to filter by  
    processes          -- will be used for multiprocessing, it essentially translates to cpus
    force_update       -- if True will drop the Redis DB and assume a new train 
    verbose            -- if True will output to console
    """
   
    now = time.time()

    if not verbose: #no output
        logger.setLevel(0)

    man = RedisManager(force_update=force_update)
    
    man.r.set('training_sample_count', train_samples) #set this for testing offsets later

    if 'classifier' in man.r.keys():
        logger.info("Trained classifier exists in Redis.")
        return

    logger.info('Storing %d feature samples.' % word_count_samples)
    man.store_word_counts(word_count_samples, processes=processes)

    logger.info('Build frequency distributions with %d features.' % word_count_range)
    man.build_freqdists(word_count_range)
    
    logger.info('Storing feature scores.')
    man.store_word_scores()
    
    logger.info('Storing %d most informative features.' % bestwords_to_store)
    man.store_best_words(bestwords_to_store)

    samples = get_samples(train_samples)

    best_words = man.get_best_words()
    trainfeats = []

    logger.info('Building feature set with %d samples.' % train_samples)
    for text, label in samples:
        s_text = sanitize_text(text)
        tokens = feat_ex(s_text, best_words=best_words)

        if tokens: trainfeats.append((tokens, label))
    
    if not trainfeats:
        logger.error( "Could not produce a training feature set.")
        return

    logger.info('Built feature set.')
    
    logger.info('Train on %d instances' % len(trainfeats))
    classifier = NaiveBayesClassifier.train(trainfeats)
    logger.info('Done training')
    
    man.store_classifier(classifier)
    logger.info('Stored to Redis')

    logger.info("Finished in: %s seconds." % (time.time() - now,))

#References: http://streamhacker.com/
#            http://text-processing.com/
#def example_train(feat_ex):
#    from nltk.corpus import movie_reviews
#    import collections
#    import nltk.metrics
#    import cPickle as pickle
#
#    negids = movie_reviews.fileids('neg')
#    posids = movie_reviews.fileids('pos')
#
#    negfeats = [(feat_ex(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
#    posfeats = [(feat_ex(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
#
#    negcutoff = len(negfeats)*3/4 #3/4 training set rest testing set
#    poscutoff = len(negfeats)*3/4
#
#    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
#    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
#    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
#
#    classifier = NaiveBayesClassifier.train(trainfeats)
#
#    refsets = collections.defaultdict(set)
#    testsets = collections.defaultdict(set)
#
#    for i, (feats, label) in enumerate(testfeats):
#        refsets[label].add(i)
#        observed = classifier.classify(feats)
#        testsets[observed].add(i)
#
#
#    print '#### POSITIVE ####'
#    print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
#    print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
#    print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
#    print
#    print '#### NEGATIVE ####'
#    print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
#    print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
#    print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])
#
#    print '--------------------'
#    print 'Classifier Accuracy:', util.accuracy(classifier, testfeats)
#    classifier.show_most_informative_features()

if __name__ == "__main__":
    #example train and tester.test to display accuracies
    from tester import test
    
    train(
        train_samples=50000,
        word_count_samples=20000,
        word_count_range=15000,
        bestwords_to_store = 5000,
        force_update=True,
        verbose=True
    )
    
    test()
