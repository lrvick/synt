# -*- coding: utf-8 -*-
from nltk.classify import NaiveBayesClassifier, util
from utils.redis_manager import RedisManager
from synt.utils.extractors import best_word_feats
from synt.utils.db import get_samples
from synt.utils.text import sanitize_text
from synt.logger import create_logger

def train(feat_ex=best_word_feats, train_samples=400000, wordcount_samples=300000, \
    wordcount_range=150000, force_update=False, verbose=True):
    """
    Trains a Naive Bayes classifier with samples from database and stores the 
    resulting classifier in Redis.
  
    Args:
    featx             -- the feature extractor to use, found in utils/extractors.py

    Keyword arguments:
    train_samples     -- the amount of samples to train half this number will be negative the other positive 
    wordcount_samples -- the amount of samples to build wordcounts, this produces a word:count histogram in Redis 
    wordcount_range   -- the amount of 'up-to' words to use for the FreqDist will pick out the most
                         'popular' words up to this amount. i.e top 150000 tokens 
    force_update      -- if True will drop the Redis DB and assume a new train 
    verbose           -- if True will output to console
    """
    
    logger = create_logger(__file__)
    if not verbose: #no output
        logger.setLevel(0)

    man = RedisManager(force_update=force_update)

    if 'classifier' in man.r.keys():
        logger.info("Trained classifier exists in Redis.")
        return

    logger.info('Storing %d word counts.' % wordcount_samples)
    man.store_word_counts(wordcount_samples)
    logger.info('Build frequency distributions with %d words.' % wordcount_range)
    man.build_freqdists(wordcount_range)
    logger.info('Storing word scores.')
    man.store_word_scores()
    logger.info('Storing best words.')
    man.store_best_words()

    samples = get_samples(train_samples)

    half = len(samples) / 2
    
    pos_samples = samples[:half]
    neg_samples = samples[half:]
   
    logger.info('Build negfeats and posfeats')
    negfeats, posfeats = [], []

    for text, sent in neg_samples:
        s_text = sanitize_text(text)
        tokens = feat_ex(s_text)
        
        if tokens:
            negfeats.append((tokens,sent))
    
    for text, sent in pos_samples:
        s_text = sanitize_text(text)
        tokens = feat_ex(s_text) 
        
        if tokens:
            posfeats.append((tokens,sent))
   
    if not (negfeats or posfeats):
        logger.error( "Could not build positive and negative features.")
        return

    negcutoff = len(negfeats)*3/4 # 3/4 training set
    poscutoff = len(posfeats)*3/4 

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats  = negfeats[negcutoff:] + posfeats[poscutoff:]
    logger.info('Train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

    classifier = NaiveBayesClassifier.train(trainfeats)
    logger.info('Done training')
    
    man.store_classifier(classifier)
    logger.info('Stored to Redis')

#   refsets = collections.defaultdict(set)
#   testsets = collections.defaultdict(set)

#   for i, (feats, label) in enumerate(testfeats):
#       if feats:
#           refsets[label].add(i)
#           observed = classifier.classify(feats)
#           testsets[observed].add(i)
#
#   print '#### POSITIVE ####'
#   print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
#   print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
#   print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
#   print
#   print '#### NEGATIVE ####'
#   print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
#   print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
#   print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

    #print '--------------------'
    logger.info('Classifier Accuracy: %s' % util.accuracy(classifier, testfeats))


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
    train(
        train_samples=2000,
        wordcount_samples=1000,
        wordcount_range=2000,
        force_update=False,
        verbose=True
    )
    from tester import test
    test()

