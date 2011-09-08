from nltk.classify import NaiveBayesClassifier, util
from collections import defaultdict
from utils.redis_manager import RedisManager
from synt.utils.extractors import best_word_feats
from synt.utils.db import get_samples
from synt.utils.text import sanitize_text
import nltk.metrics

def train(feat_ex, train_samples=400000, wordcount_samples=300000, \
    wordcount_range=150000, force_update=False):
    """
    Trains a Naive Bayes classifier with samples from database and stores it in redis.
    
    Force update will force a new train.
    """
   
    man = RedisManager(force_update=force_update)

    if 'classifier' in man.r.keys():
        print("Trained classifier exists in Redis.")
        return

    print('Storing %d word counts.' % wordcount_samples)
    man.store_word_counts(wordcount_samples)
    print('Build frequency distributions with %d words.' % wordcount_range)
    man.build_freqdists(wordcount_range)
    print('Storing word scores.')
    man.store_word_scores()

    samples = get_samples(train_samples)
    
    half = train_samples / 2

    neg_samples = samples[half:]
    pos_samples = samples[:half]
    
    negfeats, posfeats = [], []
    for text, sent in neg_samples:
        tokens = feat_ex(sanitize_text(text))
        if tokens:
            negfeats.append((tokens,sent))
    
    for text, sent in pos_samples:
        tokens = feat_ex(sanitize_text(text))
        if tokens:
            posfeats.append((tokens,sent))
    
    print('Build negfeats and posfeats')

    negcutoff = len(negfeats)*3/4 # 3/4 training set
    poscutoff = len(posfeats)*3/4 

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats  = negfeats[negcutoff:] + posfeats[poscutoff:]
    print('Train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

    classifier = NaiveBayesClassifier.train(trainfeats)

    print('Done training ...')
    
    man.store_classifier(classifier)
    print('Stored to Redis ...')

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
    print('Classifier Accuracy:', util.accuracy(classifier, testfeats))
    classifier.show_most_informative_features(50)


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
    train(best_word_feats, force_update=False)
