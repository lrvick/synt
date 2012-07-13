# -*- coding: utf-8 -*-
from nltk import FreqDist, ELEProbDist
from utils.db import RedisManager, get_samples, db_exists
from collections import defaultdict
from synt.utils.extractors import get_extractor
from synt import config

def train(db_name, samples=200000, classifier_type='naivebayes', extractor_type='words',
    best_features=10000, processes=8, purge=False):
    """
    Train with samples from sqlite database and stores the resulting classifier in Redis.

    Arguments:
    db_name (str) -- Name of the training database to use stored in ~/.synt

    Keyword arguments:
    samples (int) -- Amount of samples to train on.
    classifier_type (str) -- Type of classifier to use. Available classifiers are 'naivebayes'.
    extractor_type (str) -- Type of extractor to use. Available extractors are 'words', 'stopwords', 'bestwords'.
    best_features (int) -- Amount of highly informative features to store.
    processes (int) -- The amount of processes to be used for counting features in parallel.
    purge (bool) -- If true will flush the redis database.
    """
    m = RedisManager(purge=purge)

    extractor = get_extractor(extractor_type)

    if not db_exists(db_name):
        raise ValueError("Database '%s' does not exist." % db_name)

    if classifier_type in m.r.keys():
        print("Classifier exists in Redis. Purge to re-train.")
        return

    classifier = config.CLASSIFIERS.get(classifier_type)
    if not classifier: #classifier not supported
        raise ValueError("Classifier '%s' not supported." % classifier_type)

    #retrieve training samples from database
    train_samples = get_samples(db_name, samples)

    m.store_feature_counts(train_samples, processes=processes)
    m.store_feature_scores()

    if best_features and best_features > 1:
        m.store_best_features(best_features)

    label_freqdist = FreqDist()
    feature_freqdist = defaultdict(FreqDist)

    #retreieve the actual samples processed for label
    neg_processed, pos_processed = m.r.get('negative_processed'), m.r.get('positive_processed')
    label_freqdist.inc('negative', int(neg_processed))
    label_freqdist.inc('positive', int(pos_processed))

    labeled_feature_freqs = m.pickle_load('labeled_feature_freqs')
    labels = labeled_feature_freqs.keys()

    #feature extraction
    feat_ex = extractor()
    extracted_set = set([feat_ex.extract(labeled_feature_freqs[label].keys(), as_list=True) for label in labels][0])

    #increment the amount of times a given feature for label occured and fill in the missing occurences with Falses
    for label in labels:
        samples = label_freqdist[label]
        for fname in extracted_set:
            trues = labeled_feature_freqs[label].get(fname, 0)
            falses = samples - trues
            feature_freqdist[label, fname].inc(True, trues)
            feature_freqdist[label, fname].inc(False, falses)

    #create the P(label) distribution
    estimator = ELEProbDist
    label_probdist = estimator(label_freqdist)

    #create the P(fval|label, fname) distribution
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = estimator(freqdist, bins=2)
        feature_probdist[label,fname] = probdist

    #TODO: naivebayes supports this prototype, future classifiers will most likely not
    trained_classifier = classifier(label_probdist, feature_probdist)

    m.pickle_store(classifier_type, trained_classifier)
    m.r.set('trained_to', samples)
    m.r.set('trained_db', db_name)
    m.r.set('trained_classifier', classifier_type)
    m.r.set('trained_extractor', extractor_type)

if __name__ == "__main__":
    #example train
    import time

    db_name       = 'samples.db'
    samples       = 10000
    best_features = 5000
    processes     = 8
    purge         = True
    extractor     = 'words'

    print("Beginning train on {} samples using '{}' db..".format(samples, db_name))
    start = time.time()
    train(
            db_name       = db_name,
            samples       = samples,
            best_features = best_features,
            extractor_type= extractor,
            processes     = processes,
            purge         = purge,
    )
    print("Successfully trained in {} seconds.".format(time.time() - start))
