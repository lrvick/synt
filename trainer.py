from nltk.classify import NaiveBayesClassifier, util

from collections import defaultdict
from utils import RedisManager, sanitize_text, best_bigram_word_feats, get_samples

from nltk.corpus import movie_reviews
import collections
import nltk.metrics
import cPickle as pickle
import datetime

def train(num_samples=500, stepby=1000):
    """
    Mimicks the train method of the NaiveBayesClassiffier but
    stores it to a peristent Redis datastore.
    """

    r = redis.Redis()
    r.flushdb()
    print("Flushed Redis DB")

    labels = ['negative','positive']
    samples_left = num_samples
    offset = 0
    while samples_left > 0:
        if samples_left > stepby:
            samples_set = stepby
            samples_left -= stepby
            offset = num_samples - samples_left
        else:
            samples_set = samples_left
            samples_left = 0
        print samples_left,num_samples
        labeled_featuresets = top_tokens(samples_set,offset)
        if labeled_featuresets:
            feature_freqdist = defaultdict(FreqDist)
            fnames = set()
            for featureset, label in labeled_featuresets:
                for fname, fval in featureset.items():
                    feature_freqdist[label, fname].inc(fval)
                    fnames.add(fname)
            for label in labels:
                for fname in fnames:
                    count = feature_freqdist[label, fname].N()
                    if count > 0:
                        prev_score = r.zscore(label, fname)
                        r.zadd(label, fname, count if not prev_score else count + prev_score)
                        print "Label: %s | Fname: %s | Count: %s" %(label,fname,count)

def trainv2(feat_ex, num_samples=200000):
    r = RedisManager()

    samples = get_samples(num_samples)

    half = num_samples / 2

    neg_samples = samples[half:]
    pos_samples = samples[:half]
    print 'neg samples %s, pos samples %s' % (len(neg_samples), len(pos_samples))
    
    #negfeats = [(feat_ex(sanitize_text(text)), sent) for (text,sent) in neg_samples if text]
    #posfeats = [(feat_ex(sanitize_text(text)), sent) for (text,sent) in pos_samples if text]
    
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
    classifier.show_most_informative_features(30)


def test_train(feat_ex):
    from nltk.corpus import movie_reviews
    import collections
    import nltk.metrics
    import cPickle as pickle

    r = redis.Redis()

    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(feat_ex(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(feat_ex(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = len(negfeats)*3/4 #3/4 training set rest testing set
    poscutoff = len(negfeats)*3/4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    #if r.exists('classifier'):
    #    print "Loading classifier from Redis ..."
    #    classifier = pickle.loads(r.get('classifier'))
    #else:
    classifier = NaiveBayesClassifier.train(trainfeats)
    #print "Saving classifier to Redis ..."
    #dump = pickle.dumps(classifier)
    #r.set('classifier', dump)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)


    print '#### POSITIVE ####'
    print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
    print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
    print 'pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
    print
    print '#### NEGATIVE ####'
    print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
    print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])
    print 'neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

    print '--------------------'
    print 'Classifier Accuracy:', util.accuracy(classifier, testfeats)
    classifier.show_most_informative_features()


def collect_samples():
    neg_lastid = None
    pos_lastid = None
    while True:
        time.sleep(1)
        pos_lastid = twitter_feed('positive',pos_lastid)
        neg_lastid = twitter_feed('negative',neg_lastid)


if __name__ == "__main__":
    from utils import word_feats, stopword_word_feats, bigram_word_feats,  best_bigram_word_feats

    print("Training ...")
    trainv2(bigram_word_feats, num_samples=100000)
    #print 'Word feats without stopwords'
    #test_train(word_feats)
    #print 'Word feats with stopwords'
    #test_train(stopword_word_feats)
    #print 'Word feats with bigrams'
    #test_train(bigram_word_feats)

    #train(200000, 10000)
