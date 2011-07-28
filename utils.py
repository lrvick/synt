from nltk.probability import FreqDist
from redis import Redis

class RedisFreqDist(FreqDist):
        '''Should work just like an nltk.probability.FreqDist, but stores freqs in
        a redis db instead of an in-memory dictionary. FreqDists otherwise can get
        quite big.
        >>> rfd = RedisFreqDist()
        >>> rfd.clear()
        >>> rfd.N()
        0
        >>> rfd.B()
        0
        >>> rfd.keys()
        []
        >>> rfd.samples()
        []
        >>> rfd.items()
        []
        >>> 'foo' in rfd
        False
        >>> rfd.inc('foo')
        >>> 'foo' in rfd
        True
        >>> rfd['foo']
        1
        >>> rfd.N()
        1
        >>> rfd.B()
        1
        >>> rfd.items()
        [('foo', 1)]
        >>> rfd.inc('zaz', 2)
        >>> rfd.samples()
        ['foo', 'zaz']
        >>> rfd.keys()
        ['zaz', 'foo']
        >>> rfd.values()
        [2, 1]
        >>> rfd.items()
        [('zaz', 2), ('foo', 1)]
        >>> rfd.N()
        3
        >>> rfd.B()
        2
        >>> rfd.clear()
        '''
        sampleskey = '__samples__'
        
        def __init__(self, samples=None, r=None, host='localhost', port=6379, db=0):
                '''Create a redis backed FreqDist. Can take an existing Redis object
                or a host and port to create a new Redis object.
                '''
                if not r:
                        r = Redis(host, port, db=db)
                self.r = r
                self._len_cache = None
                FreqDist.__init__(self, samples)
                # NOTE: not calling clear so can create a RedisFreqDist with existing data
        
        ## dict methods ##
        
        def __getitem__(self, sample):
                assert sample != self.sampleskey
                self.r.select(self.r.db)
                count = self.r.get(sample)
                if count: return int(count)
                else: return None
        
        def __setitem__(self, sample, count):
                assert sample != self.sampleskey
                self.r.select(self.r.db)
                self.r.set(sample, count)
                self.r.sadd(self.sampleskey, sample)
                self._invalidate()
        
        def __delitem__(self, sample):
                assert sample != self.sampleskey
                self.r.select(self.r.db)
                self.r.srem(self.sampleskey, sample)
                self.r.delete(sample)
                self._invalidate()
        
        def __contains__(self, sample):
                return self.r.sismember(self.sampleskey, sample)
        
        def __len__(self):
                # TODO: use SCARD when available in redis.py
                if not self._len_cache:
                        self._len_cache = len(self.keys())
                return self._len_cache
        
        def clear(self):
                self.r.select(self.r.db)
                
                for sample in self.samples():
                        self.r.delete(sample)
                
                self.r.delete(self.sampleskey)
                self._invalidate()
        
        def keys(self):
                self.r.select(self.r.db)
                
                if self.r.exists(self.sampleskey):
                        return self.r.sort(self.sampleskey, by='*', desc=True)
                else:
                        return []
        
        def values(self):
                return list(self.itervalues())
        
        def items(self):
                return list(self.iteritems())
        
        def itervalues(self):
                for sample in self.iterkeys():
                        yield self[sample]
        
        def iteritems(self):
                for sample in self.iterkeys():
                        yield sample, self[sample]
        
        ## FreqDist methods ##
        
        def _invalidate(self):
                self._len_cache = None
                # internal to FreqDist
                self._Nr_cache = None
                self._N = None
                self._max_cache = None
        
        def samples(self):
                self.r.select(self.r.db)
                
                if self.r.exists(self.sampleskey):
                        return self.r.sort(self.sampleskey, alpha=True)
                else:
                        return []
        
        def N(self):
                if not self._N:
                        self._N = sum(count for count in self.itervalues())
                return self._N
        
        def inc(self, sample, count=1):
                assert sample != self.sampleskey
                self.r.select(self.r.db)
                self.r.sadd(self.sampleskey, sample)
                
                if count > 0:
                        self.r.incr(sample, count)
                else:
                        self.r.decr(sample, count)
                
                self._invalidate()
        
        def freq(self, sample):
                if sample in self:
                        self.N() # make sure have accurate self._N
                        return FreqDist.freq(self, sample)
                else:
                        return 0.0

if __name__=="__main__":

    def run_test(redis=True):
        from collections import defaultdict
        from nltk.probability import FreqDist
        if redis == False:
            label_freqdist = FreqDist()
            feature_freqdist = defaultdict(FreqDist)
        else:
            label_freqdist = RedisFreqDist()
            feature_freqdist = defaultdict(RedisFreqDist)
     
        labeled_featuresets = [
            ({'neg_word1': True, 'neg_word2': True, 'neg_word3': True}, u'negative'),
            ({'neg_word4': True, 'neg_word5': True, 'neg_word6': True}, u'negative'),
            ({'neg_word1': True, 'neg_word2': True, 'neg_word3': True}, u'negative'),
            ({'neg_word4': True, 'neg_word5': True, 'neg_word6': True}, u'negative'),
            ({'pos_word1': True, 'pos_word2': True, 'pos_word3': True}, u'positive'),
            ({'pos_word4': True, 'pos_word5': True, 'pos_word6': True}, u'positive'),
            ({'pos_word1': True, 'pos_word2': True, 'pos_word3': True}, u'positive'),
            ({'pos_word4': True, 'pos_word5': True, 'pos_word6': True}, u'positive'),
        ]
        feature_values = defaultdict(set)
        fnames = set()

        for featureset, label in labeled_featuresets:
            label_freqdist.inc(label) 
            for fname, fval in featureset.items():
                feature_freqdist[label, fname].inc(fval) 
                feature_values[fname].add(fval) 
                fnames.add(fname)
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                feature_freqdist[label, fname].inc(None, num_samples-count)
                feature_values[fname].add(None)

        return label_freqdist,feature_freqdist,feature_values
    
    import doctest
    print doctest.testmod()
    
    print '\n\n-------Without Redis-------\n'    
    label_freqdist,feature_freqdist,feature_values = run_test(False)
    print '%s \n' % label_freqdist
    print '%s \n' % feature_values
    print '%s \n' % feature_freqdist
   
    print '\n\n-------With Redis----------\n' 
    label_freqdist,feature_freqdist,feature_values = run_test(True)
    print '%s \n' % label_freqdist
    print '%s \n' % feature_values
    print '%s \n' % feature_freqdist
