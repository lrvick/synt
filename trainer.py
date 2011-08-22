import os
import re
import sqlite3
import nltk
from redis import Redis
from nltk.classify import NaiveBayesClassifier
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist
from nltk.tokenize.treebank import TreebankWordTokenizer
from collections import defaultdict
from utils import top_tokens, bag_of_words, sanitize_text, db_init
import settings


def train(num_samples=500, stepby=1000):
    """
    Mimicks the train method of the NaiveBayesClassiffier but
    stores it to a peristent Redis datastore.
    """

    r = Redis()
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
        labeled_featuresets = get_tokens(samples_set,offset)
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


def collect_samples():
    neg_lastid = None
    pos_lastid = None
    while True:
        time.sleep(1)
        pos_lastid = twitter_feed('positive',pos_lastid)
        neg_lastid = twitter_feed('negative',neg_lastid)
