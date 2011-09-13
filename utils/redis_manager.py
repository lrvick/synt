"""Tools for interacting with Redis"""

import ast
import redis
import itertools
import cPickle as pickle
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from synt.utils.text import sanitize_text
from synt.utils.db import get_samples

class RedisManager(object):

    def __init__(self, force_update=False):
        """
        Initializes redis. If force_update is true
        will flush the database and assume an initial setup.
        """
        
        self.r = redis.Redis()
        self.force_update = force_update
        if self.force_update:
            self.r.flushdb()


    def build_freqdists(self, wordcount_range=150000):
        """
        Build word and label freq dists from the stored words with n words
        and store the resulting FreqDists in Redis.

        This cannot be cached as we have to continously update these values
        from incremented word counts.
        """

        word_freqdist = FreqDist()
        label_word_freqdist = ConditionalFreqDist()

        pos_words = self.r.zrange('positive_wordcounts', 0, wordcount_range, withscores=True, desc=True)
        neg_words = self.r.zrange('negative_wordcounts', 0, wordcount_range, withscores=True, desc=True)

        assert pos_words and neg_words, 'Requires wordcounts to be stored in redis.'

        for word,count in pos_words:
            word_freqdist.inc(word, count=count)
            label_word_freqdist['pos'].inc(word, count=count)

        for word,count in neg_words:
            word_freqdist.inc(word, count=count)
            label_word_freqdist['neg'].inc(word, count=count)

        #storing for use later, these values are always computed
        self.r.set('word_fd', pickle.dumps(word_freqdist))
        self.r.set('label_fd', pickle.dumps(label_word_freqdist))
        

    def store_word_counts(self, wordcount_samples=300000):
        """
        Stores word counts for label in Redis with the ability to increment.
        """

        if 'positive_wordcounts' and 'negative_wordcounts' in self.r.keys():
            return
       
        samples = get_samples(wordcount_samples)
        assert samples, "Samples must be provided."

        for text, label in samples:
            label = label + '_wordcounts'
            tokens = sanitize_text(text)

            if tokens:
                for word in tokens:
                    prev_score = self.r.zscore(label, word)
                    self.r.zadd(label, word, 1 if not prev_score else prev_score + 1)

    def store_word_scores(self):
        """
        Stores 'word scores' into Redis.
        
        """
        
        try:
            word_freqdist = pickle.loads(self.r.get('word_fd'))
            label_word_freqdist = pickle.loads(self.r.get('label_fd'))
        except TypeError:
            print('Requires frequency distributions to be built.')

        word_scores = {}

        pos_word_count = label_word_freqdist['pos'].N()
        neg_word_count = label_word_freqdist['neg'].N()
        total_word_count = pos_word_count + neg_word_count

        for word, freq in word_freqdist.iteritems():
            pos_score = BigramAssocMeasures.chi_sq(label_word_freqdist['pos'][word], (freq, pos_word_count), total_word_count)

            neg_score = BigramAssocMeasures.chi_sq(label_word_freqdist['neg'][word], (freq, neg_word_count), total_word_count)

            word_scores[word] = pos_score + neg_score
        
        self.r.set('word_scores', word_scores)


    def store_classifier(self, classifier, name='classifier'):
        """
        Stores a pickled a classifier into Redis.
        """
        dumped = pickle.dumps(classifier, protocol=1)
        self.r.set(name, dumped)
        

    def load_classifier(self, name='classifier'):
        """
        Loads (unpickles) a classifier from Redis.
        """
        try:
            loaded = pickle.loads(self.r.get(name))
        except TypeError:
            return     
        return loaded 

    def get_top_words(self, label, start=0, end=10):
        """Return the top words for label from Redis store."""
        if self.r.exists(label):
            return self.r.zrange(label, start, end, withscores=True, desc=True) 

    def store_best_words(self, n=10000):
        """Store n best words to Redis."""

        word_scores = ast.literal_eval(self.r.get('word_scores')) #str -> dict
            
        assert word_scores, "Word scores need to exist."

        best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:n]
        self.r.set('best_words', best)
        

    def get_best_words(self, scores=False):
        """
        Return cached best_words

        If scores provided will return word/score tuple.
        """

        if 'best_words' in self.r.keys():
            best_words =  ast.literal_eval(self.r.get('best_words'))

            if not scores:
                best_words = [w[0] for w in best_words]

            return best_words
