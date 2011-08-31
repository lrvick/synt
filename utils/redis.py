"""Tools for interacting with Redis"""

import redis
import itertools
import cPickle as pickle
from nltk.probability import FreqDist, ConditionalFreqDist

class RedisManager(object):

    def __init__(self):
        """
        Initializes redis.
        """
        
        self.r = redis.Redis()
    
    def build_freqdists(self, n=300000):
        """ Build word and label freq dists from the stored words with n words. """

        word_freqdist = FreqDist()
        label_word_freqdist = ConditionalFreqDist()

        pos_words = self.r.zrange('positive_wordcounts', 0, n, withscores=True, desc=True)
        neg_words = self.r.zrange('negative_wordcounts', 0, n, withscores=True, desc=True)

        for word,count in pos_words:
            word_freqdist.inc(word, count=count)
            label_word_freqdist['pos'].inc(word, count=count)

        for word,count in neg_words:
            word_freqdist.inc(word, count=count)
            label_word_freqdist['neg'].inc(word, count=count)

        return (word_freqdist, label_word_freqdist)


    def store_wordcounts(self, samples):
        """
        Stores word counts for label in Redis with the ability to increment.
        
        Expects a list of samples in the form (text, sentiment) 
        ex. (u'This is a text string', 'neg')
        """

        if samples:
            
            for text,label in samples:
                label = label + '_wordcounts'
                tokens = sanitize_text(text)

                if tokens:
                    for word in tokens:
                        prev_score = self.r.zscore(label, word)
                   
                        self.r.zadd(label, word, 1 if not prev_score else prev_score + 1)

    def store_word_scores(self, word_freqdist, label_word_freqdist):
        """
        Stores 'word scores' into Redis.
        """
        
        word_scores = {}

        pos_word_count = label_word_freqdist['pos'].N()
        neg_word_count = label_word_freqdist['neg'].N()
        total_word_count = pos_word_count + neg_word_count

        for word, freq in word_freqdist.iteritems():
            print (word, freq)
            pos_score = BigramAssocMeasures.chi_sq(label_word_freqdist['pos'][word], (freq, pos_word_count), total_word_count)

            neg_score = BigramAssocMeasures.chi_sq(label_word_freqdist['neg'][word], (freq, neg_word_count), total_word_count)

            word_scores[word] = pos_score + neg_score
        
        self.r.set('word_scores', word_scores)


    def store_classifier(self, classifier, name='classifier'):
        """
        Stores a pickled a classifier into Redis.
        """
        dumped = pickle.dumps(classifier, protocol=pickle.HIGHEST_PROTOCOL)
        self.r.set(name, dumped)
        

    def load_classifier(self, name='classifier'):
        """
        Loads (unpickles) a classifier from Redis.
        """
        loaded = pickle.loads(self.r.get(name))
        return loaded 

    def top_words(self, label, start=0, end=10):
        """Return the top words for label from Redis store."""
        if self.r.exists(label):
            return self.r.zrange(label, start, end, withscores=True, desc=True) 

    def best_words(self, n=10000):
        """Return n best words."""
        import ast

        try:
            word_scores = ast.literal_eval(self.r.get('word_scores')) #str -> dict
        except ValueError:
            raise ValueError('Malformed string, make sure its a proper dictionary.')
            

        if word_scores:
            best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:n]
            words = set([w for w,s in best])
            return words


