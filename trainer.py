from nltk.classify import NaiveBayesClassifier, util
from collections import defaultdict
from synt.utils,redis_manager import RedisManager
from synt.utils.extractors import sanitize_text, best_word_feats
from synt.utils.db import get_samples
import nltk.metrics
import cPickle as pickle
import datetime

def train(feat_ex, num_samples=200000):
    """
    Trains a Naive Bayes classifier with samples from database and stores it in redis.
    """
    
    r = RedisManager()

    samples = get_samples(num_samples)

    half = num_samples / 2

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

    
    negcutoff = len(negfeats)* 3/4 # 3/4 training set
    poscutoff = len(posfeats)* 3/4 

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats  = negfeats[negcutoff:] + posfeats[poscutoff:]
    print
    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    classifier = NaiveBayesClassifier.train(trainfeats)
    
    print 'Done training ...'
    print 'Storing to redis ...'
    r.r.set('classifier', pickle.dumps(classifier))
    print 'Stored.'

    #r.load_classifier(feats=trainfeats, force_train=True)    


#    refsets = collections.defaultdict(set)
#    testsets = collections.defaultdict(set)
#
#    for i, (feats, label) in enumerate(testfeats):
#        if feats:
#            refsets[label].add(i)
#            observed = classifier.classify(feats)
#            testsets[observed].add(i)
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
    print '--------------------'
    print 'Classifier Accuracy:', util.accuracy(classifier, testfeats)
    classifier.show_most_informative_features(50)


#References: http://streamhacker.com/
#            http://text-processing.com/
#def example_train(feat_ex):
#    from nltk.corpus import movie_reviews
#    import collections
#    import nltk.metrics
#    import cPickle as pickle
#
#    r = redis.Redis()
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
    
    train(best_word_feats, num_samples=200000)
