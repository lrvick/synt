# -*- coding: utf-8 -*-
from nltk import NaiveBayesClassifier, FreqDist, ELEProbDist
from utils.db import RedisManager
from synt.logger import create_logger
import time
import cPickle as pickle
from collections import defaultdict

logger = create_logger(__file__)

def train(samples=200000, best_features=10000, processes=8):
    """
    Trains a Naive Bayes classifier with samples from database and stores the 
    resulting classifier in Redis.
  
    Keyword arguments:
    samples         -- the amount of samples to train on
    best_features   -- amount of highly informative features to store
    processes       -- will be used for multiprocessing, it essentially translates to cpus
    """
   
    start = time.time()

    m = RedisManager()
    m.r.set('training_sample_count', samples)

    if 'classifier' in m.r.keys():
        logger.info("Trained classifier exists in Redis. Purge first to re-train.")
        return

    logger.info('Storing feature counts for %d samples.' % samples)
    m.store_feature_counts(samples, processes=processes)

    logger.info('Building frequency distributions from feature counts.')
    m.store_freqdists()
    
    logger.info('Storing feature scores.')
    m.store_feature_scores()
    
    if best_features:
        logger.info('Storing %d most informative features.' % best_features)
        m.store_best_features(best_features)

    label_freqdist = FreqDist()
    feature_freqdist = defaultdict(FreqDist)
    feature_values = defaultdict(set)
    fnames = set()

    neg_processed, pos_processed = m.r.get('negative_processed'), m.r.get('positive_processed')
    label_freqdist.inc('negative', int(neg_processed))
    label_freqdist.inc('positive', int(pos_processed))

    conditional_fd = pickle.loads(m.r.get('label_fd'))
    
    labels = conditional_fd.conditions()

    for label in labels:
        for fname, fcount in conditional_fd[label].items():
            feature_freqdist[label, fname].inc(True, fcount)
            feature_values[fname].add(True)
            fnames.add(fname)
   
    for label in labels:
        num_samples = label_freqdist[label] #sample count for label 
        for fname in fnames:
            count = feature_freqdist[label, fname].N()
            feature_freqdist[label, fname].inc(None, num_samples - count)
            feature_values[fname].add(None)

    # Create the P(label) distribution
    estimator = ELEProbDist
    label_probdist = estimator(label_freqdist)
    
    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = estimator(freqdist, bins=len(feature_values[fname]))
        feature_probdist[label,fname] = probdist
    
    logger.info('Built feature probdist with %d items.' % len(feature_probdist.items()))

    classifier = NaiveBayesClassifier(label_probdist, feature_probdist)
    logger.info('Initialized classifier.')
    
    m.store_classifier(classifier)
    logger.info('Stored classifier to Redis.')
    
    logger.info('Finished in: %s seconds.' % (time.time() - start))

if __name__ == "__main__":
    #example train

    train(
        samples       = 100,
        best_features = None,
        processes     = 8,
    )
