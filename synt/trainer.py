# -*- coding: utf-8 -*-
from nltk import FreqDist, ELEProbDist
from utils.db import RedisManager, get_samples, db_exists
from collections import defaultdict
from synt.utils.extractors import get_extractor
from synt import config

def train(db, samples=200000, classifier='naivebayes', extractor='words', best_features=10000, processes=8, purge=False, redis_db=5):
    """
    Train with samples from sqlite database and stores the resulting classifier in Redis.

    Arguments:
    db              -- the name of the training database to use stored in ~/.synt

    Keyword arguments:
    samples         -- the amount of samples to train on
    classifier      -- the type of classifier to use NOTE: currently only naivebayes is supported
    extractor       -- the extractor to use, available extraots are 'words', 'stopwords', 'bestwords'
    best_features   -- amount of highly informative features to store
    processes       -- will be used for counting features in parallel 
    redis_db        -- the redis database to use 
    """
   
    _extractor = get_extractor(extractor)
    
    if not (db_exists(db) or samples <= 0):
        return

    m = RedisManager(db=redis_db, purge=purge) 

    if classifier in m.r.keys():
        print("Classifier exists in Redis. Purge to re-train.")
        return

    _classifier = config.CLASSIFIERS.get(classifier, None)
    if not _classifier: #not supported 
        return

    #get training samples from database 
    train_samples = get_samples(db, samples)

    m.store_feature_counts(train_samples, processes=processes)
    m.store_freqdists()
    m.store_feature_scores()
   
    if best_features:
        m.store_best_features(best_features)

    label_freqdist = FreqDist()
    feature_freqdist = defaultdict(FreqDist)

    neg_processed, pos_processed = m.r.get('negative_processed'), m.r.get('positive_processed')
    label_freqdist.inc('negative', int(neg_processed))
    label_freqdist.inc('positive', int(pos_processed))

    conditional_fd = m.pickle_load('label_fd')

    labels = conditional_fd.conditions()
    
    #feature extraction
    feat_ex = _extractor()
    extracted_set = set([feat_ex.extract(conditional_fd[label].keys(), as_list=True) for label in labels][0])  

    for label in labels:
        samples = label_freqdist[label]
        for fname in extracted_set:
            trues = conditional_fd[label][fname] #is the count it happened
            falses = samples - trues
            feature_freqdist[label, fname].inc(True, trues)
            feature_freqdist[label, fname].inc(False, falses)

    # Create the P(label) distribution
    estimator = ELEProbDist
    label_probdist = estimator(label_freqdist)
    
    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = estimator(freqdist, bins=2) 
        feature_probdist[label,fname] = probdist
    
    #TODO: naivebayes supports this prototype, future classifiers will most likely not
    #support various classifiers
    _c = _classifier(label_probdist, feature_probdist) 
    
    m.pickle_store(classifier, _c)
    m.r.set('trained_to', samples)
    m.r.set('trained_db', db)
    m.r.set('trained_classifier', classifier)
    m.r.set('trained_extractor', extractor)

if __name__ == "__main__":
    #example train
    import time

    db            = 'samples.db'
    samples       = 10000
    best_features = 5000 
    processes     = 8
    purge         = True
    extractor     = 'words' 
    redis_db      = 5

    print("Beginning train on {} samples using '{}' db..".format(samples, db))
    start = time.time()
    train(
            db            = db, 
            samples       = samples,
            best_features = best_features,
            extractor     = extractor,
            processes     = processes,
            purge         = purge,
            redis_db      = redis_db,
    )
    print("Successfully trained in {} seconds.".format(time.time() - start))
