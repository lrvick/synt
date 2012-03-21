# -*- coding: utf-8 -*-
"""Tools to interact with databases."""

import os
import sqlite3
import redis
import cPickle as pickle
#from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.metrics import BigramAssocMeasures
from synt.utils.text import normalize_text
from synt.utils.processing import batch_job
from synt import config

def db_exists(name):
    """
    Returns true if the database exists in our path defined by SYNT_PATH.

    Arguments:
    name (str) -- Database name.

    """
    path = os.path.join(config.SYNT_PATH, name)
    return True if os.path.exists(path) else False

def db_init(db, create=True):
    """
    Initializes the sqlite3 database.

    Keyword Arguments:
    db (str) -- Name of the database to use.
    create (bool) -- If creating the database for the first time.

    """
    
    if not os.path.exists(config.SYNT_PATH):
        os.makedirs(config.SYNT_PATH)

    fp = os.path.join(config.SYNT_PATH, db)

    if not db_exists(db):
        conn = sqlite3.connect(fp)
        cursor = conn.cursor()
        if create:
            cursor.execute('''CREATE TABLE item (id integer primary key, text text unique, sentiment text)''')
    else:
        conn = sqlite3.connect(fp)
    return conn


def redis_feature_consumer(samples, **kwargs):
    """
    Stores feature and counts to redis via a pipeline.
    """

    if 'db' not in kwargs:
        raise KeyError("Feature consumer requires db.")

    db = kwargs['db']

    rm = RedisManager(db=db)
    pipeline = rm.r.pipeline()

    neg_processed, pos_processed = 0, 0

    for text, label in samples:

        count_label = label + '_feature_counts'

        tokens = normalize_text(text)

        if tokens:
            if label.startswith('pos'):
                pos_processed += 1
            else:
                neg_processed += 1

            for word in set(tokens): #make sure we only add word once
                pipeline.zincrby(count_label, word)

    pipeline.incr('negative_processed', neg_processed)
    pipeline.incr('positive_processed', pos_processed)

    pipeline.execute()

class RedisManager(object):

    def __init__(self, db=config.REDIS_DB, host=config.REDIS_HOST, password=config.REDIS_PASSWORD, purge=False):
        self.r = redis.Redis(db=db, host=host, password=password)
        self.db = db
        if purge is True:
            self.r.flushdb()

    def store_feature_counts(self, samples, chunksize=10000, processes=None):
        """
        Stores feature:count histograms for samples in Redis with the ability to increment.

        Arguments:
        samples (list) -- List of samples in the format (text, label)

        Keyword Arguments:
        chunksize (int) -- Amount of samples to process at a time.
        processes (int) -- Amount of processors to use with multiprocessing.

        """

        if 'positive_feature_counts' and 'negative_feature_counts' in self.r.keys():
            return

        #do this with multiprocessing
        c_args = {'db': self.db}
        batch_job(samples, redis_feature_consumer, chunksize=chunksize, processes=processes, consumer_args=c_args)

    def store_feature_scores(self):
        """
        Build scores based on chi-sq and store from stored features then save their scores to Redis.
        """
        
        pos_words = self.r.zrange('positive_feature_counts', 0, -1, withscores=True, desc=True)
        neg_words = self.r.zrange('negative_feature_counts', 0, -1, withscores=True, desc=True)

        assert pos_words and neg_words, 'Requires feature counts to be stored in redis.'

        feature_freqs = {}
        labeled_feature_freqs = {'positive': {}, 'negative': {}}
        labels = labeled_feature_freqs.keys() 

        #build a condtional freqdist with the feature counts per label
        for feature,freq in pos_words:
            feature_freqs[feature] = freq
            labeled_feature_freqs['positive'].update({feature : freq})

        for feature,freq in neg_words:
            feature_freqs[feature] = freq 
            labeled_feature_freqs['negative'].update({feature : freq})

        scores = {}

        pos_feature_count = len(labeled_feature_freqs['positive'])
        neg_feature_count = len(labeled_feature_freqs['negative'])
        total_feature_count = pos_feature_count + neg_feature_count

        for label in labels:
            for feature,freq in feature_freqs.items():
                pos_score = BigramAssocMeasures.chi_sq(
                        labeled_feature_freqs['positive'].get(feature, 0),
                        (freq, pos_feature_count),
                        total_feature_count
                )
                neg_score = BigramAssocMeasures.chi_sq(
                        labeled_feature_freqs['negative'].get(feature, 0),
                        (freq, neg_feature_count),
                        total_feature_count
                )

                scores[feature] = pos_score + neg_score

        self.pickle_store('feature_freqs', feature_freqs)
        self.pickle_store('labeled_feature_freqs', labeled_feature_freqs)
        self.pickle_store('scores', scores)

    def store_best_features(self, n=10000):
        """
        Stores the best features in Redis.

        Keyword Arguments:
        n (int) -- Amount of features to store as best features.

        """
        if not n: return

        feature_scores = self.pickle_load('scores')

        assert feature_scores, "Feature scores need to exist."

        best = sorted(feature_scores.items(), key=lambda (w,s): s, reverse=True)[:n]

        self.pickle_store('best_features',  best)

    def get_best_features(self):
        """
        Return stored best features.
        """
        best_features = self.pickle_load('best_features')

        if best_features:
            return set([feature for feature,score in best_features])

    def pickle_store(self, name, data):
        dump = pickle.dumps(data, protocol=1) #highest_protocol breaks with NLTKs FreqDist
        self.r.set(name, dump)

    def pickle_load(self, name):
        return pickle.loads(self.r.get(name))

def get_sample_limit(db):
    """
    Returns the limit of samples so that both positive and negative samples
    will remain balanced.

    Keyword Arguments:
    db (str) -- Name of the database to use.

    """

    #this is an expensive operation in case of a large database
    #therefore we store the limit in redis and use that when we can
    m = RedisManager()
    if 'limit' in m.r.keys():
        return int(m.r.get('limit'))

    db = db_init(db=db)
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM item where sentiment = 'positive'")
    pos_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM item where sentiment = 'negative'")
    neg_count = cursor.fetchone()[0]
    if neg_count > pos_count:
        limit = pos_count
    else:
        limit = neg_count

    m.r.set('limit', limit)

    return limit

def get_samples(db, limit, offset=0):
    """
    Returns a combined list of negative and positive samples in a (text, label) format.

    Arguments:
    db (str) -- Name of the databse to use.
    limit (int) -- Amount of samples to retrieve.

    Keyword Arguments:
    offset (int) -- Where to start getting samples from.

    """
    conn = db_init(db=db)
    cursor = conn.cursor()

    sql =  "SELECT text, sentiment FROM item WHERE sentiment = ? LIMIT ? OFFSET ?"

    if limit < 2: limit = 2

    if limit > get_sample_limit(db):
        limit = get_sample_limit(db)

    if limit % 2 != 0:
        limit -= 1 #we want an even number

    limit = limit / 2
    offset = offset / 2

    cursor.execute(sql, ["negative", limit, offset])
    neg_samples = cursor.fetchall()

    cursor.execute(sql, ["positive", limit, offset])
    pos_samples = cursor.fetchall()

    return pos_samples + neg_samples

