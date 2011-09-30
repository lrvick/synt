# -*- coding: utf-8 -*-
"""Tools for interacting with Redis"""

import ast
import redis
import cPickle as pickle
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from synt.utils.text import sanitize_text
from synt.logger import create_logger
import multiprocessing

logger = create_logger(__file__)

def _c(samples):
    """
    Stores counts to redis via a pipeline.
    """
    
    man = RedisManager()
   
    pipeline = man.r.pipeline()
    
    for text, label in samples:
        
        label = label + '_wordcounts'
        tokens = sanitize_text(text)

        if tokens:
            for word in tokens:
                pipeline.zincrby(label, word)
        
    pipeline.execute()
    logger.info('finished %d chunked samples' % len(samples))


class RedisManager(object):

    def __init__(self, db=0, force_update=False):
        """
        Initializes redis. If force_update is true
        will flush the database and assume an initial setup.
        """
        
        self.r = redis.Redis(db=db)
        self.force_update = force_update
        if self.force_update:
            self.r.flushdb()


    def build_freqdists(self, wordcount_range=150000):
        """
        Build word and label freq dists from the stored words with 'wordcount_range' words
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
        

    def store_word_counts(self, wordcount_samples=300000, chunksize=10000, processes=None):
        """
        Stores word:count histograms for samples in Redis with the ability to increment.
        
        Keyword Arguments:
        wordcount_samples   -- the amount of samples to use for determining feature counts
        chunksize           -- the amount of samples to process at a time
        processes           -- the amount of processors to run in async
                               each process will be handed a chunksize of samples
                               i.e:
                               4 processes will be handed 10000 samples. If this is none 
                               it will be set to the default cpu count of your computer.
        """

        if 'positive_wordcounts' and 'negative_wordcounts' in self.r.keys():
            return
        
        from synt.utils.db import get_samples
      
        if not processes:
            processes = multiprocessing.cpu_count()
       
        offset = 0

        while offset != wordcount_samples:
           
            pool = multiprocessing.Pool(processes)
            logger.info("Spawning %d processes." % processes)

            for i in range(1, processes + 1): #for each process
                
                if offset >= wordcount_samples: 
                    #if our offset has reached the sample count, break out
                    break
              
                if offset + chunksize > wordcount_samples:
                    #if our offset and chunksize is greater than samples we have
                    #subtract samples and offset to get our remainder.
                    #ex:
                    # offset = 900, chunksize = 300, samples = 1000
                    # offset + chunksize = 1200 (greater)
                    # samples - offset = 100 remainder
                    chunksize = wordcount_samples - offset

                samples = get_samples(chunksize, offset=offset)

                pool.apply_async(_c, [samples,]) #give chunks to workers
   
                offset += chunksize 

            pool.close()
            pool.join() #wait for workers to finish


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
                #in case of no scores we don't care about order
                tmp = {}
                for w in best_words:
                    tmp.setdefault(w[0], None)
                best_words = tmp 
                #best_words = [w[0] for w in best_words]

            return best_words
