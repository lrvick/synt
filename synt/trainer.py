# -*- coding: utf-8 -*-
from nltk import NaiveBayesClassifier, FreqDist, ELEProbDist
from utils.db import RedisManager
from collections import defaultdict

def train(samples=200000, classifier='naivebayes', best_features=10000, processes=8, purge=False):
    """
    Train with samples from sqlite database and stores the resulting classifier in Redis.
  
    Keyword arguments:
    samples         -- the amount of samples to train on
    classifier      -- the classifier to use 
                       NOTE: currently only naivebayes is supported
    best_features   -- amount of highly informative features to store
    processes       -- will be used for counting features in parallel 
    """
   
    m = RedisManager(purge=purge)
    m.r.set('training_sample_count', samples)

    if classifier in m.r.keys():
        print("Classifier exists in Redis. Purge to re-train.")
        return

    m.store_feature_counts(samples, processes=processes)
    m.store_freqdists()
    m.store_feature_scores()
    
    if best_features:
        m.store_best_features(best_features)

    label_freqdist = FreqDist()
    feature_freqdist = defaultdict(FreqDist)
    feature_values = defaultdict(set)
    fnames = set()

    neg_processed, pos_processed = m.r.get('negative_processed'), m.r.get('positive_processed')
    label_freqdist.inc('negative', int(neg_processed))
    label_freqdist.inc('positive', int(pos_processed))

    conditional_fd = m.pickle_load('label_fd')
    
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
    
    c = NaiveBayesClassifier(label_probdist, feature_probdist)
    
    #TODO: support various classifiers
    m.store_classifier(classifier, c)

if __name__ == "__main__":
    #example train

    train(
        samples       = 10000,
        best_features = None,
        processes     = 8,
    )
